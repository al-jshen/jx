import h5py
import argparse
from tqdm.auto import tqdm


def main():
    with open(args.in_file) as f:
        filenames = [i.strip() for i in f.readlines()]

    total_size = 0

    chunks = [h5py.File(f, "r") for f in filenames]

    cat = h5py.File(args.out_file, "w")

    for f in chunks:
        k = list(f.keys())[0]
        total_size += f[k].shape[0]

    for k in chunks[0].keys():
        shape = list(chunks[0][k].shape)
        shape[0] = total_size
        shape = tuple(shape)
        cat.create_dataset(
            k,
            shape=shape,
            dtype=chunks[0][k].dtype,
            chunks=True,
            compression="lzf",
            shuffle=True,
        )

    counter = 0
    for f in tqdm(chunks):
        for k in tqdm(f.keys()):
            curr_size = f[k].shape[0]
            cat[k][counter : counter + curr_size] = f[k][:]
        counter += curr_size
        f.close()

    cat.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join H5 catalogs into a single file")
    parser.add_argument(
        "--in-file", type=str, help="Name of file containing input file names"
    )
    parser.add_argument("--out-file", type=str, help="Name of output file")
    args = parser.parse_args()
    main()
