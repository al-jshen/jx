import argparse

import h5py
import numpy as np


def load_velocities(f):

    source_ids = f.get("source_id")[:]

    pmra = f.get("pmra")[:].astype("float32")
    pmra_error = f.get("pmra_error")[:].astype("float32")

    pmdec = f.get("pmdec")[:].astype("float32")
    pmdec_error = f.get("pmdec_error")[:].astype("float32")

    rv = f.get("radial_velocity")[:].astype("float32")
    rv_error = f.get("radial_velocity_error")[:].astype("float32")

    mask = np.logical_or.reduce(
        (
            np.isnan(pmra),
            np.isnan(pmdec),
            np.isnan(rv),
            np.isnan(pmra_error),
            np.isnan(pmdec_error),
            np.isnan(rv_error),
        )
    )

    return (
        source_ids[mask],
        pmra[mask],
        pmra_error[mask],
        pmdec[mask],
        pmdec_error[mask],
        rv[mask],
        rv_error[mask],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract XP spectra coefficients.")
    parser.add_argument(
        "--in-file", type=str, help="File containing list of HDF5 files to process"
    )
    parser.add_argument("--out-file", type=str, help="Output file")
    args = parser.parse_args()

    with open(args.in_file, "r") as f:
        filenames = [i.strip() for i in f.readlines()]

    results = dict(
        source_id=[],
        pmra=[],
        pmra_error=[],
        pmdec=[],
        pmdec_error=[],
        rv=[],
        rv_error=[],
    )

    for i, f in enumerate(filenames):
        with h5py.File(f, "r") as f_curr:
            (
                source_id,
                pmra,
                pmra_error,
                pmdec,
                pmdec_error,
                rv,
                rv_error,
            ) = load_velocities(f_curr)

            results["source_id"].append(source_id),
            results["pmra"].append(pmra),
            results["pmra_error"].append(pmra_error),
            results["pmdec"].append(pmdec),
            results["pmdec_error"].append(pmdec_error),
            results["rv"].append(rv),
            results["rv_error"].append(rv_error)

    results = {k: np.stack(v) for k, v in results.items()}

    all_len = results["source_id"].size

    with h5py.File(args.out_file, "a") as f:
        f.create_dataset(
            "source_id",
            shape=(all_len,),
            dtype="int64",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["source_id"],
        )
        f.create_dataset(
            "pmra",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["pmra"],
        )
        f.create_dataset(
            "pmra_error",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["pmra_error"],
        )
        f.create_dataset(
            "pmdec",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["pmdec"],
        )
        f.create_dataset(
            "pmdec_error",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["pmdec_error"],
        )
        f.create_dataset(
            "radial_velocity",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["rv"],
        )
        f.create_dataset(
            "radial_velocity_error",
            shape=(all_len,),
            dtype="float32",
            chunks=True,
            compression="lzf",
            shuffle=True,
            data=results["rv_error"],
        )
