import numpy as np
import h5py
import argparse
import time
import astropy.units as u
from astropy.coordinates import SkyCoord
from dustmaps import sfd


def main():

    start = time.time()

    linebreak = "==============================================================="

    print("Adding RA, Dec, and parallaxes")
    print(linebreak)

    catalog = h5py.File(f"{args.to_file}", "r+")
    rdp = h5py.File(f"{args.data_file}", "r")

    end_idx = rdp["source_id"].size if args.end_idx == -1 else args.end_idx

    full_size = end_idx - args.start_idx

    size = catalog["source_id"].size

    assert abs(full_size - size) < 1e-3 * full_size

    match, idx1, idx2 = np.intersect1d(
        catalog["source_id"][:],
        rdp["source_id"][args.start_idx : end_idx],
        return_indices=True,
    )

    assert match.size == size
    assert np.all(
        np.isclose(
            catalog["source_id"][:][idx1],
            rdp["source_id"][args.start_idx : end_idx][idx2],
        )
    )

    ra = rdp["ra"][args.start_idx : end_idx][idx2]
    dec = rdp["dec"][args.start_idx : end_idx][idx2]

    cat_args = dict(chunks=True, compression="lzf", shuffle=True)

    catalog.create_dataset(
        "ra",
        shape=(size,),
        data=ra,
        **cat_args,
    )
    catalog.create_dataset(
        "dec",
        shape=(size,),
        data=dec,
        **cat_args,
    )
    catalog.create_dataset(
        "parallax",
        shape=(size,),
        dtype="float32",
        data=rdp["parallax"][args.start_idx : end_idx][idx2].astype("float32"),
        **cat_args,
    )
    catalog.create_dataset(
        "parallax_error",
        shape=(size,),
        dtype="float32",
        data=rdp["parallax_error"][args.start_idx : end_idx][idx2],
        **cat_args,
    )

    print("Getting dust maps")
    print(linebreak)

    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
    )

    rdp.close()

    sfdquery = sfd.SFDQuery()
    ebv = sfdquery.query(coords)

    catalog.create_dataset(
        "sfd_ebv",
        shape=(size,),
        dtype="float32",
        data=ebv.astype("float32"),
        **cat_args,
    )

    catalog.close()

    end = time.time()

    print(f"Done, took {end - start:.2f} seconds")
    print(linebreak)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add positions and EBV to catalog")
    parser.add_argument(
        "--to-file", help="Add values to this HDF5 file", type=str, required=True
    )
    parser.add_argument(
        "--data-file",
        help="HDF5 file containing positions and source IDs to cross match against",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Start index of catalog to process"
    )
    parser.add_argument(
        "--end-idx", type=int, default=-1, help="End index of catalog to process"
    )

    args = parser.parse_args()
    main()
