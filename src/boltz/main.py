import glob
import pickle
import shutil
import string
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from rdkit import Chem
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1.ckpt"


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str], msa_dir: Path, msa_server_url: str, msa_pairing_strategy: str
) -> list[Path]:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    msa_dir : Path
        The msa temp directory.

    Returns
    -------
    list[Path]
        The list of MSA files.

    """
    # Run MMSeqs2
    msa = run_mmseqs2(
        list(data.values()),
        msa_dir,
        use_pairing=len(data) > 1,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    # Dump to A3M
    for idx, key in enumerate(data):
        entity_msa = msa[idx]
        msa_path = msa_dir / f"{key}.a3m"
        with msa_path.open("w") as f:
            f.write(entity_msa)


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA seuqneces, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Parse input data
    records: list[Record] = []
    for path in tqdm(data):
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            target = parse_yaml(path, ccd)
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                entity_id = chain.entity_id
                msa_id = f"{target_id}_{entity_id}"
                to_generate[msa_id] = target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.a3m"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                chain.msa_id = -1

        # Generate MSA
        if to_generate and not use_msa_server:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        if to_generate:
            msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            click.echo(msg)
            compute_msa(
                to_generate,
                msa_dir,
                msa_server_url=msa_server_url,
                msa_pairing_strategy=msa_pairing_strategy,
            )

        # Parse MSA data
        msas = {c.msa_id for c in target.record.chains if c.msa_id != -1}
        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            if not msa_path.exists():
                msg = f"MSA file {msa_path} not found."
                raise FileNotFoundError(msg)

            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
            if not processed.exists():
                msa: MSA = parse_a3m(
                    msa_path,
                    taxonomy=None,
                    max_seqs=max_msa_seqs,
                )
                msa.dump(processed)

        # Modify records to point to processed MSA
        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz1."""
    return


# Define a reusable decorator for screen and predict
def shared_options(func):
    func = click.option(
        "--out_dir",
        type=click.Path(exists=False),
        default="./",
        help="The path where to save the predictions.",
    )(func)
    func = click.option(
        "--cache",
        type=click.Path(exists=False),
        default="~/.boltz",
        help="The directory where to download the data and model. Default is ~/.boltz.",
    )(func)
    func = click.option(
        "--checkpoint",
        type=click.Path(exists=True),
        default=None,
        help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    )(func)
    func = click.option(
        "--devices",
        type=int,
        default=1,
        help="The number of devices to use for prediction. Default is 1.",
    )(func)
    func = click.option(
        "--accelerator",
        type=click.Choice(["gpu", "cpu", "tpu"]),
        default="gpu",
        help="The accelerator to use for prediction. Default is gpu.",
    )(func)
    func = click.option(
        "--recycling_steps",
        type=int,
        default=3,
        help="The number of recycling steps to use for prediction. Default is 3.",
    )(func)
    func = click.option(
        "--sampling_steps",
        type=int,
        default=200,
        help="The number of sampling steps to use for prediction. Default is 200.",
    )(func)
    func = click.option(
        "--diffusion_samples",
        type=int,
        default=1,
        help="The number of diffusion samples to use for prediction. Default is 1.",
    )(func)
    func = click.option(
        "--output_format",
        type=click.Choice(["pdb", "mmcif"]),
        default="mmcif",
        help="The output format to use for the predictions. Default is mmcif.",
    )(func)
    func = click.option(
        "--num_workers",
        type=int,
        default=2,
        help="The number of dataloader workers to use for prediction. Default is 2.",
    )(func)
    func = click.option(
        "--override",
        is_flag=True,
        help="Whether to override existing found predictions. Default is False.",
    )(func)
    func = click.option(
        "--use_msa_server",
        is_flag=True,
        help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
    )(func)
    func = click.option(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MSA server url. Used only if --use_msa_server is set.",
    )(func)
    func = click.option(
        "--msa_pairing_strategy",
        type=str,
        default="greedy",
        help="Pairing strategy to use. Options are 'greedy' and 'complete'.",
    )(func)
    return func


# Predict workflow
def predict_input(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
    }
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(BoltzDiffusionParams()),
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@shared_options
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    predict_input(
        data,
        out_dir=out_dir,
        cache=cache,
        checkpoint=checkpoint,
        devices=devices,
        accelerator=accelerator,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        output_format=output_format,
        num_workers=num_workers,
        override=override,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )


# small function to parse a complete SDF to smiles
def _process_sdf(sdf_path: str):
    output_dict = {}
    suppl = Chem.SDMolSupplier(sdf_path)

    for mol in suppl:
        if mol is not None:
            mol_smiles = Chem.MolToSmiles(mol)
            if mol.HasProp("_Name"):
                mol_name = mol.GetProp("_Name")
                if mol_name == "":
                    mol_name = mol_smiles
            else:
                mol_name = mol_smiles

            output_dict[mol_name] = mol_smiles

    return output_dict


@cli.command()
@click.option(
    "--protein",
    type=click.Path(exists=True),
    required=True,
    help="The path to the PDB or fasta file",
)
@click.option(
    "--ligands",
    type=click.Path(exists=True),
    required=True,
    help=(
        "Path to the compounds to screen against your protein. This can be either: "
        "a directory containing multiple SDF files, a single SDF file with multiple structures, "
        "or a text file with compound IDs and their corresponding SMILES strings (in that order)."
    ),
)
@click.option(
    "--msa_path",
    type=click.Path(exists=False),
    help="The path to precomputed MSA (should be in m3a format)",
    default="",
)
@shared_options
def screen(
    protein: str,
    ligands: str,
    msa_path: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Screen many ligands against 1 protein target with Boltz-1."""
    protein_path = Path(protein).expanduser()
    ligand_path = Path(ligands).expanduser()

    # Process the protein input
    protein_name = protein_path.stem

    if protein_path.suffix.lower() == ".pdb":
        # Get FASTA sequence from pdb file using biopython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", protein_path)
        ppb = PPBuilder()

        protein_seq = {
            chain.id: str(pp.get_sequence())
            for model in structure
            for chain in model
            for pp in ppb.build_peptides(chain)
        }

    elif protein_path.suffix.lower() in (".fa", ".fas", ".fasta"):
        # Process as fasta file
        with protein_path.open("r") as f:
            protein_seq = {
                string.ascii_uppercase[i]: str(record.seq)
                for i, record in enumerate(SeqIO.parse(protein_path, "fasta"))
            }

    else:
        msg = f"File format {path.suffix} not supported, please provide file in pdb or fasta format"
        raise click.ClickException(msg)

    # Get a list of all the ligand smiles
    smiles_dict = {}

    if ligand_path.is_file():
        ligand_name = ligand_path.stem

        # check the extension
        if ligand_path.suffix.lower() == ".sdf":
            smiles_dict.update(_process_sdf(ligand_path))
        elif ligand_path.suffix.lower() in [".smi", ".ism", ".smiles"]:
            # split and add to dict
            with open(ligand_path) as ligand_file:
                ligand_lines = ligand_file.readlines()
                for line in ligand_lines:
                    line_split = line.strip().split()
                    smiles_dict[line_split[0]] = line_split[1]
        else:
            msg = f"Files with {ligand_path.suffix} extension are not supported as ligand. Only .sdf and .smi files are supported"
            raise click.ClickException(msg)

    else:
        ligand_name = ligand_path.name

        # Get all the sdf files and add them to the dictionary
        ligand_files = glob.glob(f"{ligand_path}/*.sdf")

        for ligand_file in ligand_files:
            smiles_dict.update(_process_sdf(ligand_file))

    msg = f"Succesfully identified {len(smiles_dict)} ligands."
    click.echo(msg)

    # Generate output directory
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{protein_name}_{ligand_name}"

    # Check if the output directory already exists
    if out_dir.exists():
        click.echo(f"The output directory '{out_dir}' already exists.")
        if not override:
            click.confirm(
                "Do you want to delete the existing directory and continue?",
                abort=True,
            )
            # Delete the directory if confirmed
            shutil.rmtree(out_dir)
            click.echo()
        else:
            click.echo("Override flag is set. The existing directory will be used.")

    # Create the output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = out_dir / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    # Perform the alignment MSA alignment using the FASTA sequence (if use_msa_server is given and no path is given)
    if msa_path == "" and use_msa_server == True:
        msg = f"Generating MSA for {protein_name}."
        click.echo(msg)
        compute_msa(
            protein_seq,
            msa_dir,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
        )
    else:
        msg = "No MSA path given, and use_msa_server is set to false. Please generate the MSA manually and include it as an argument, or consider adding --use_msa_server if the protein sequence is not confidential."
        raise click.ClickException(msg)

    # Make a directory where we can write all the query files to
    query_dir = out_dir / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)

    # Make the query template
    query_template = ""

    for chain in protein_seq:
        query_template += (
            f">{chain}|protein|{msa_dir}/{chain}.a3m\n{protein_seq[chain]}\n"
        )
    query_template += f">{string.ascii_uppercase[len(protein_seq)]}|smiles\n"

    for query_id in smiles_dict:
        with open(query_dir / f"{query_id}_{protein_name}.fasta", "w") as query_file:
            query_file.write(f"{query_template}{smiles_dict[query_id]}\n")

    # Call the predict_input function, where we pass all variables
    predict_input(
        query_dir,
        out_dir=out_dir,
        cache=cache,
        checkpoint=checkpoint,
        devices=devices,
        accelerator=accelerator,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        output_format=output_format,
        num_workers=num_workers,
        override=override,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )


if __name__ == "__main__":
    cli()
