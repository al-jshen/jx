import warnings

import argparse
import time
import joblib
import astropy.units as u
import h5py
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from dustmaps import sfd
from prfr import ProbabilisticRandomForestRegressor, split_arrays
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    start_time = time.time()

    linebreak = "==============================================================="

    print("Running cross-match")
    print(linebreak)

    f = h5py.File(f"{args.data_dir}/gdr3_astronn_xp_coeffs_full.h5", "r")

    data_apogee = fits.open(f"{args.data_dir}/apogee_astroNN-DR17.fits")[1].data

    cols = [
        "ra",
        "dec",
        "TEFF",
        "TEFF_ERR",
        "LOGG",
        "LOGG_ERR",
        "C_H",
        "C_H_ERR",
        "CI_H",
        "CI_H_ERR",
        "N_H",
        "N_H_ERR",
        "O_H",
        "O_H_ERR",
        "NA_H",
        "NA_H_ERR",
        "MG_H",
        "MG_H_ERR",
        "AL_H",
        "AL_H_ERR",
        "SI_H",
        "SI_H_ERR",
        "P_H",
        "P_H_ERR",
        "S_H",
        "S_H_ERR",
        "K_H",
        "K_H_ERR",
        "CA_H",
        "CA_H_ERR",
        "TI_H",
        "TI_H_ERR",
        "TIII_H",
        "TIII_H_ERR",
        "V_H",
        "V_H_ERR",
        "CR_H",
        "CR_H_ERR",
        "MN_H",
        "MN_H_ERR",
        "FE_H",
        "FE_H_ERR",
        "CO_H",
        "CO_H_ERR",
        "NI_H",
        "NI_H_ERR",
        "age",
        "age_total_error",
    ]
    _data = {key: data_apogee[key].astype("float32") for key in cols}
    _data["source_id"] = data_apogee["source_id"].astype("int64")
    df_apogee = (
        pd.DataFrame(_data)
        .dropna(
            subset=[
                "ra",
                "dec",
                "TEFF",
                "TEFF_ERR",
                "LOGG",
                "LOGG_ERR",
                "MG_H",
                "MG_H_ERR",
                "FE_H",
                "FE_H_ERR",
                "age",
                "age_total_error",
            ]
        )
        .drop_duplicates(subset="source_id")
    )

    print("Making data cuts")
    print(linebreak)

    mask = np.logical_and.reduce(
        (
            df_apogee["FE_H"] > -5.0,
            df_apogee["FE_H"] < 5.0,
            df_apogee["MG_H"] > -5.0,
            df_apogee["MG_H"] < 5.0,
            df_apogee["LOGG"] > 0.0,
            df_apogee["LOGG"] < 7.0,
            df_apogee["TEFF"] > 0.0,
            df_apogee["TEFF"] < 50_000.0,
            df_apogee["age"] > 0.0,
            df_apogee["age"] < 14.0,
            df_apogee["FE_H_ERR"] < 0.5,
            df_apogee["MG_H_ERR"] < 0.5,
            df_apogee["LOGG_ERR"] < 0.5,
            np.abs(df_apogee["TEFF_ERR"] / df_apogee["TEFF"]) < 0.5,
            # np.abs(df_apogee["age_total_error"] / df_apogee["age"]) < 1.0,
            np.isfinite(df_apogee["TEFF"]),
            np.isfinite(df_apogee["LOGG"]),
            np.isfinite(df_apogee["FE_H"]),
            np.isfinite(df_apogee["MG_H"]),
            np.isfinite(df_apogee["TEFF"]),
            np.isfinite(df_apogee["age"]),
        )
    )

    df_apogee = df_apogee[mask]

    print("Extracting features and labels")
    print(linebreak)

    matches, idx_apogee_full, idx_xp_full = np.intersect1d(
        df_apogee["source_id"], f["ids"][:], return_indices=True
    )

    features = (
        f["coeffs"][:][idx_xp_full] / f["phot_g_mean_flux"][:][idx_xp_full][:, None]
    )
    efeatures = np.abs(features) * np.sqrt(
        (f["coeff_errs"][:][idx_xp_full] / f["coeffs"][:][idx_xp_full]) ** 2
        + (
            (
                f["phot_g_mean_flux_error"][:][idx_xp_full]
                / f["phot_g_mean_flux"][:][idx_xp_full]
            )
            ** 2
        )[:, None]
    )

    f.close()

    label_names = ["FE_H", "MG_H", "LOGG", "TEFF", "age"]
    elabel_names = ["FE_H_ERR", "MG_H_ERR", "LOGG_ERR", "TEFF_ERR", "age_total_error"]
    labels = df_apogee.iloc[idx_apogee_full][label_names].to_numpy()
    elabels = df_apogee.iloc[idx_apogee_full][elabel_names].to_numpy()
    labels[:, 1] = labels[:, 1] - labels[:, 0]  # mg/h - fe/h = mg/fe
    elabels[:, 1] = np.sqrt(elabels[:, 0] ** 2 + elabels[:, 1] ** 2)

    train, test, valid = split_arrays(
        features,
        labels,
        efeatures,
        elabels,
        test_size=0.2 if args.test else int(1),
        valid_size=0.2,
    )

    if args.load_prfr_model != "":
        print(f"Loading PRFR model from {args.load_prfr_model}")
        print(linebreak)
        model = joblib.load(args.load_prfr_model)
    else:
        print(f"Training PRFR model on {len(train[0])} stars")
        print(linebreak)

        model = ProbabilisticRandomForestRegressor(n_jobs=-1)
        model.fit(train[0], train[1], eX=train[2], eY=train[3])
        model.calibrate(valid[0], valid[1], eX=valid[2], eY=valid[3], apply_bias=False)
        model.fit_bias(valid[0], valid[1], eX=valid[2])

        if args.save_prfr_model != "":
            print(f"Saving PRFR model to {args.save_prfr_model}")
            print(linebreak)
            joblib.dump(model, args.save_prfr_model)

    if args.test:
        print("Evaluating PRFR model performance")
        print(linebreak)

        preds, biases = model.predict(test[0], eX=test[2], return_bias=True)
        residual = test[1] - np.mean(preds, axis=-1)

        with np.printoptions(precision=4, suppress=True):
            print(f"Bias: {residual.mean(axis=0)}")
            print(f"Scatter: {residual.std(axis=0)}")

    if not args.test_only:

        if args.load_knn_model != "":
            print(f"Loading KNN model from {args.load_knn_model}")
            print(linebreak)

            knn = joblib.load(args.load_knn_model)
            dists = joblib.load(f"{args.load_knn_model}.dists")

        else:
            print("Training KNN model")
            print(linebreak)

            knn = NearestNeighbors(n_neighbors=args.knn_nneighbours + 1, n_jobs=12)
            knn.fit(train[0])

            print("Determining KNN distance threshold")
            print(linebreak)

            dists, _ = knn.kneighbors(
                train[0], n_neighbors=args.knn_nneighbours + 1, return_distance=True
            )
            dists = dists[:, -1]

            if args.save_knn_model != "":
                print(f"Saving KNN model to {args.save_knn_model}")
                print(linebreak)

                joblib.dump(knn, args.save_knn_model)
                joblib.dump(dists, f"{args.save_knn_model}.dists")

        xs = np.sort(dists)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        thresh = np.interp(args.knn_threshold, ys, xs)

        print(f"Creating catalog file at {args.out_file}")
        print(linebreak)

        xp = h5py.File(f"{args.data_dir}/gdr3_xp_coeffs.h5")

        catalog = h5py.File(args.out_file, "w")

        end_idx = xp["ids"].size if args.end_idx == -1 else args.end_idx
        full_size = end_idx - args.start_idx
        chunk_counter = args.start_idx
        catalog_size = 0

        print("Determining catalog size")
        print(linebreak)

        for i in tqdm(range(np.ceil(full_size / args.chunk_size).astype(int))):
            upper = np.minimum(end_idx, chunk_counter + args.chunk_size).astype(int)

            if chunk_counter > upper:
                break

            feature = (
                xp["coeffs"][chunk_counter:upper]
                / xp["phot_g_mean_flux"][chunk_counter:upper][:, None]
            )
            nanmask = ~np.any(np.isnan(feature), axis=1)

            catalog_size += nanmask.sum()

            chunk_counter = upper

        print(f"Catalog size: {catalog_size}")
        print(linebreak)

        chunk_counter = args.start_idx
        write_counter = 0

        cat_args = dict(chunks=True, compression="lzf", shuffle=True)

        catalog.create_dataset(
            "source_id", shape=(catalog_size,), dtype="int64", **cat_args
        )

        catalog.create_dataset(
            "feh", shape=(catalog_size, 5), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "mgfe", shape=(catalog_size, 5), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "logg", shape=(catalog_size, 5), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "teff", shape=(catalog_size, 5), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "age", shape=(catalog_size, 5), dtype="float32", **cat_args
        )

        catalog.create_dataset(
            "feh_bias", shape=(catalog_size,), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "mgfe_bias", shape=(catalog_size,), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "logg_bias", shape=(catalog_size,), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "teff_bias", shape=(catalog_size,), dtype="float32", **cat_args
        )
        catalog.create_dataset(
            "age_bias", shape=(catalog_size,), dtype="float32", **cat_args
        )

        catalog.create_dataset(
            "flags", shape=(catalog_size,), dtype="int64", **cat_args
        )
        catalog.create_dataset(
            "distances", shape=(catalog_size,), dtype="float32", **cat_args
        )

        bitmask = 2 ** np.arange(7)[::-1]

        print("Generating catalog")
        print(linebreak)

        for _ in tqdm(range(np.ceil(full_size / args.chunk_size).astype(int))):
            try:
                upper = np.minimum(end_idx, chunk_counter + args.chunk_size).astype(int)

                if chunk_counter > upper:
                    break

                feature = (
                    xp["coeffs"][chunk_counter:upper]
                    / xp["phot_g_mean_flux"][chunk_counter:upper][:, None]
                )
                efeature = np.abs(feature) * np.sqrt(
                    (
                        xp["coeff_errs"][chunk_counter:upper]
                        / xp["coeffs"][chunk_counter:upper]
                    )
                    ** 2
                    + (
                        (
                            xp["phot_g_mean_flux_error"][chunk_counter:upper]
                            / xp["phot_g_mean_flux"][chunk_counter:upper]
                        )
                        ** 2
                    )[:, None]
                )
                nanmask = ~np.any(np.isnan(feature), axis=1)

                curr_len = nanmask.sum()

                preds, biases = model.predict(
                    feature[nanmask],
                    eX=efeature[nanmask],
                    return_bias=True,
                )
                distances, _ = knn.kneighbors(
                    feature[nanmask],
                    n_neighbors=args.knn_nneighbours,
                    return_distance=True,
                )
                distances = distances[:, -1]

                bias_flag = np.abs((biases / preds.std(axis=-1))) > args.bias_threshold

                too_far_flag = (distances > thresh).astype(bool)
                err_collapse_flag = np.any(
                    np.isclose(
                        np.diff(np.quantile(preds, [0.16, 0.5, 0.84], axis=-1), axis=0),
                        0.0,
                    ),
                    axis=(0, 2),
                )

                flags = np.hstack(
                    (
                        bias_flag,
                        too_far_flag.reshape(-1, 1),
                        err_collapse_flag.reshape(-1, 1),
                    )
                )

                flags_bitmask = (flags * bitmask).sum(axis=1)

                pred_qtls = np.quantile(
                    preds, [0.025, 0.16, 0.5, 0.84, 0.975], axis=2
                ).T

                write_upper = write_counter + curr_len
                catalog["source_id"][write_counter:write_upper] = xp["ids"][
                    chunk_counter:upper
                ][nanmask]

                catalog["feh"][write_counter:write_upper] = pred_qtls[0]
                catalog["mgfe"][write_counter:write_upper] = pred_qtls[1]
                catalog["logg"][write_counter:write_upper] = pred_qtls[2]
                catalog["teff"][write_counter:write_upper] = pred_qtls[3]
                catalog["age"][write_counter:write_upper] = pred_qtls[4]

                catalog["feh_bias"][write_counter:write_upper] = biases[:, 0]
                catalog["mgfe_bias"][write_counter:write_upper] = biases[:, 1]
                catalog["logg_bias"][write_counter:write_upper] = biases[:, 2]
                catalog["teff_bias"][write_counter:write_upper] = biases[:, 3]
                catalog["age_bias"][write_counter:write_upper] = biases[:, 4]

                catalog["flags"][write_counter:write_upper] = flags_bitmask
                catalog["distances"][write_counter:write_upper] = distances

                chunk_counter = upper
                write_counter = write_upper

            except Exception as e:
                print(e)
                break

        xp.close()

        print("Adding RA, Dec, and parallaxes")
        print(linebreak)

        rdp = h5py.File(f"{args.data_dir}/radecplx.h5", "r")

        match, idx1, idx2 = np.intersect1d(
            catalog["source_id"][:], rdp["source_id"][:], return_indices=True
        )

        ra = rdp["ra"][:][idx2]
        dec = rdp["dec"][:][idx2]

        catalog.create_dataset(
            "ra",
            shape=(catalog_size,),
            data=ra,
            **cat_args,
        )
        catalog.create_dataset(
            "dec",
            shape=(catalog_size,),
            data=dec,
            **cat_args,
        )
        catalog.create_dataset(
            "parallax",
            shape=(catalog_size,),
            dtype="float32",
            data=rdp["parallax"][:][idx2].astype("float32"),
            **cat_args,
        )
        catalog.create_dataset(
            "parallax_error",
            shape=(catalog_size,),
            dtype="float32",
            data=rdp["parallax_error"][:][idx2],
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
            shape=(catalog_size,),
            dtype="float32",
            data=ebv.astype("float32"),
            **cat_args,
        )

        catalog.close()

        end_time = time.time()
        print(
            f"Done. Took {end_time - start_time:.2f} seconds to process {full_size} stars. Final catalog size: {catalog_size} stars."
        )
        print(linebreak)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate full DR3 stellar parameters catalog"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Size of chunk to process at a time",
    )
    parser.add_argument(
        "--start-idx", type=int, default=0, help="Start index of catalog to process"
    )
    parser.add_argument(
        "--end-idx", type=int, default=-1, help="End index of catalog to process"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory where data files are stored"
    )
    parser.add_argument("--out-file", type=str, help="Name of output file")
    parser.add_argument(
        "--knn-threshold",
        type=float,
        default=0.95,
        help="Threshold for KNN distance flag",
    )
    parser.add_argument(
        "--knn-nneighbours",
        type=int,
        default=5,
        help="Number of neighbours for KNN distance flag",
    )
    parser.add_argument(
        "--bias-threshold",
        type=float,
        default=3.0,
        help="Threshold for bias correction flag",
    )
    parser.add_argument(
        "--save-prfr-model", type=str, default="", help="Path to save PRFR model to"
    )
    parser.add_argument(
        "--save-knn-model", type=str, default="", help="Path to save KNN model to"
    )
    parser.add_argument(
        "--load-prfr-model", type=str, default="", help="Path to load PRFR model from"
    )
    parser.add_argument(
        "--load-knn-model", type=str, default="", help="Path to load KNN model from"
    )
    parser.add_argument(
        "--test",
        help="Use 20%% of the dataset to evaluate model performance",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
    )
    parser.add_argument(
        "--test-only",
        help="Only train and evaluate PRFR model, don't make catalog",
        type=str2bool,
        default=False,
        const=True,
        nargs="?",
    )
    args = parser.parse_args()
    main()
