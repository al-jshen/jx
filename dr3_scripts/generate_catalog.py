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


def main():
    start_time = time.time()

    linebreak = "==============================================================="

    print("Running cross-match")
    print(linebreak)

    f = h5py.File("/st/datasets/gdr3_astronn_xp_coeffs_full.h5", "r")

    data_apogee = fits.open("/st/datasets/apogee_astroNN-DR17.fits")[1].data

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

    print("Getting dust maps")
    print(linebreak)

    sfdquery = sfd.SFDQuery()
    coords = SkyCoord(
        ra=df_apogee["ra"].to_numpy() * u.deg, dec=df_apogee["dec"].to_numpy() * u.deg
    )
    ebv = sfdquery(coords)
    df_apogee["sfd_ebv"] = ebv
    matches, idx_apogee_full, idx_xp_full = np.intersect1d(
        df_apogee["source_id"], f["ids"][:], return_indices=True
    )

    print("Extracting features and labels")
    print(linebreak)

    features = f["coeffs"][idx_xp_full] / f["phot_g_mean_flux"][idx_xp_full][:, None]
    labels = (
        (df_apogee["LOGG"].iloc[idx_apogee_full].to_numpy() < 3.5)
        .astype("float32")
        .reshape(-1, 1)
    )
    train, test = split_arrays(features, labels, test_size=0.2)

    if args.load_classifier_model != "":
        print(f"Loading giant classifier from {args.load_classifier_model}")
        print(linebreak)

        clf = joblib.load(args.load_classifier_model)

    else:
        print("Training giant classifier")
        print(linebreak)

        clf = RandomForestClassifier(n_jobs=-1)
        clf.fit(train[0], train[1])

        if args.save_classifier_model != "":
            print(f"Saving giant classifier to {args.save_classifier_model}")
            print(linebreak)

            joblib.dump(clf, args.save_classifier_model)

    probs = clf.predict_proba(test[0])[:, 1]
    xplot = np.linspace(0, 1, 1000)
    recalls = np.array([test[1][(probs > i)].sum() / test[1].sum() for i in xplot])
    precisions = np.array(
        [test[1][(probs > i)].sum() / ((probs > i)).sum() for i in xplot]
    )
    f1s = 2.0 * (recalls * precisions) / (recalls + precisions)
    bestidx = np.nanargmax(f1s)

    print(f"Best classifier threshold: {xplot[bestidx]}")
    print(f"Best recall: {recalls[bestidx]}")
    print(f"Best precision: {precisions[bestidx]}")

    print("Making data cuts")
    print(linebreak)

    df_apogee = df_apogee[df_apogee["FE_H"] > -5.0]
    df_apogee = df_apogee[df_apogee["LOGG"] < 3.5]
    df_apogee = df_apogee[df_apogee["LOGG_ERR"] < 0.12]
    df_apogee = df_apogee[(df_apogee["age_total_error"] / df_apogee["age"]) < 0.3]
    df_apogee = df_apogee[df_apogee["sfd_ebv"] < 0.1]

    matches, idx_apogee_cut, idx_xp_cut = np.intersect1d(
        df_apogee["source_id"], f["ids"][:], return_indices=True
    )

    print("Making new features and labels")
    print(linebreak)

    features = (
        f["coeffs"][:][idx_xp_cut] / f["phot_g_mean_flux"][:][idx_xp_cut][:, None]
    )
    efeatures = np.abs(features) * np.sqrt(
        (f["coeff_errs"][:][idx_xp_cut] / f["coeffs"][:][idx_xp_cut]) ** 2
        + (
            (
                f["phot_g_mean_flux_error"][:][idx_xp_cut]
                / f["phot_g_mean_flux"][:][idx_xp_cut]
            )
            ** 2
        )[:, None]
    )

    f.close()

    label_names = ["FE_H", "MG_H", "LOGG", "TEFF", "age"]
    elabel_names = ["FE_H_ERR", "MG_H_ERR", "LOGG_ERR", "TEFF_ERR", "age_total_error"]
    labels = df_apogee.iloc[idx_apogee_cut][label_names].to_numpy()
    elabels = df_apogee.iloc[idx_apogee_cut][elabel_names].to_numpy()
    labels[:, 1] = labels[:, 1] - labels[:, 0]  # mg/h - fe/h = mg/fe
    elabels[:, 1] = np.sqrt(elabels[:, 0] ** 2 + elabels[:, 1] ** 2)

    train, test, valid = split_arrays(
        features, labels, efeatures, elabels, test_size=0.2, valid_size=0.2
    )

    if args.load_prfr_model != "":
        print(f"Loading PRFR model from {args.load_prfr_model}")
        print(linebreak)
        model = joblib.load(args.load_prfr_model)
    else:
        print("Training PRFR model")
        print(linebreak)

        model = ProbabilisticRandomForestRegressor(n_jobs=-1)
        model.fit(train[0], train[1], eX=train[2], eY=train[3])
        model.calibrate(valid[0], valid[1], eX=valid[2], eY=valid[3], apply_bias=False)
        model.fit_bias(valid[0], valid[1], eX=valid[2])

        if args.save_prfr_model != "":
            print(f"Saving PRFR model to {args.save_prfr_model}")
            print(linebreak)
            joblib.dump(model, args.save_prfr_model)

    print("Evaluating PRFR model performance")
    print(linebreak)

    preds, biases = model.predict(test[0], eX=test[2], return_bias=True)
    residual = test[1] - np.mean(preds, axis=-1)

    with np.printoptions(precision=4, suppress=True):
        print(f"Bias: {residual.mean(axis=0)}")
        print(f"Scatter: {residual.std(axis=0)}")

    if args.load_knn_model != "":
        print(f"Loading KNN model from {args.load_knn_model}")
        print(linebreak)

        knn = joblib.load(args.load_knn_model)

    else:
        print("Training KNN model")
        print(linebreak)

        knn = NearestNeighbors(n_neighbors=args.knn_nneighbours + 1, n_jobs=32)
        knn.fit(train[0])

        if args.save_knn_model != "":
            print(f"Saving KNN model to {args.save_knn_model}")
            print(linebreak)

            joblib.dump(knn, args.save_knn_model)

    print("Determining KNN distance threshold")
    print(linebreak)

    dists_train, _ = knn.kneighbors(
        train[0], n_neighbors=args.knn_nneighbours + 1, return_distance=True
    )

    xs = np.sort(dists_train[:, -1])
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    thresh = np.interp(args.knn_threshold, ys, xs)

    print(f"Creating catalog file at {args.out_file}")
    print(linebreak)

    xp = h5py.File("/st/datasets/gdr3_xp_coeffs.h5")

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
        efeature = np.abs(feature) * np.sqrt(
            (xp["coeff_errs"][chunk_counter:upper] / xp["coeffs"][chunk_counter:upper])
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

        in_sample_prob = clf.predict_log_proba(feature[nanmask])[:, 1]
        in_sample_mask = in_sample_prob > -0.6931471805599453  # log(0.5)
        in_sample_ids = xp["ids"][chunk_counter:upper][nanmask][in_sample_mask]

        catalog_size += in_sample_ids.size

        chunk_counter = upper

    print(f"Catalog size: {catalog_size}")
    print(linebreak)

    chunk_counter = args.start_idx
    write_counter = 0

    cat_args = dict(chunks=True, compression="lzf", shuffle=True)

    catalog.create_dataset(
        "source_id", shape=(catalog_size,), dtype="int64", **cat_args
    )

    catalog.create_dataset("feh", shape=(catalog_size, 5), dtype="float32", **cat_args)
    catalog.create_dataset("mgfe", shape=(catalog_size, 5), dtype="float32", **cat_args)
    catalog.create_dataset("logg", shape=(catalog_size, 5), dtype="float32", **cat_args)
    catalog.create_dataset("teff", shape=(catalog_size, 5), dtype="float32", **cat_args)
    catalog.create_dataset("age", shape=(catalog_size, 5), dtype="float32", **cat_args)

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

    catalog.create_dataset("flags", shape=(catalog_size,), dtype="int64", **cat_args)
    catalog.create_dataset(
        "logprob_in_sample", shape=(catalog_size,), dtype="float32", **cat_args
    )
    catalog.create_dataset(
        "distances", shape=(catalog_size,), dtype="float32", **cat_args
    )

    bitmask = 2 ** np.arange(7)[::-1]

    print("Generating catalog")
    print(linebreak)

    for i in tqdm(range(np.ceil(full_size / args.chunk_size).astype(int))):
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

            in_sample_prob = clf.predict_log_proba(feature[nanmask])[:, 1]
            in_sample_mask = in_sample_prob > -0.6931471805599453  # log(0.5)
            in_sample_ids = xp["ids"][chunk_counter:upper][nanmask][in_sample_mask]

            curr_len = in_sample_ids.size

            catalog_size += curr_len

            preds, biases = model.predict(
                feature[nanmask][in_sample_mask],
                eX=efeature[nanmask][in_sample_mask],
                return_bias=True,
            )
            distances, _ = knn.kneighbors(
                feature[nanmask][in_sample_mask],
                n_neighbors=args.knn_nneighbours,
                return_distance=True,
            )
            distances = distances[:, -1]

            bias_flag = np.abs((biases / preds.std(axis=-1))) > args.bias_threshold

            too_far_flag = (distances > thresh).astype(bool)
            err_collapse_flag = np.any(
                np.isclose(
                    np.diff(np.quantile(preds, [0.16, 0.5, 0.84], axis=-1), axis=0), 0.0
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

            pred_qtls = np.quantile(preds, [0.025, 0.16, 0.5, 0.84, 0.975], axis=2).T

            curr_len = in_sample_ids.size

            write_upper = write_counter + curr_len
            catalog["source_id"][write_counter:write_upper] = in_sample_ids

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
            catalog["logprob_in_sample"][write_counter:write_upper] = in_sample_prob[
                in_sample_mask
            ]
            catalog["distances"][write_counter:write_upper] = distances

            chunk_counter = upper
            write_counter = write_upper

        except Exception as e:
            print(e)
            break

    catalog.close()
    xp.close()

    end_time = time.time()
    print(
        f"Done, took {end_time - start_time:.2f} seconds to process {full_size} stars"
    )
    print(linebreak)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate DR3 stellar parameters catalog"
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
        "--save-classifier-model",
        type=str,
        default="",
        help="Path to save classifier model to",
    )
    parser.add_argument(
        "--save-prfr-model", type=str, default="", help="Path to save PRFR model to"
    )
    parser.add_argument(
        "--save-knn-model", type=str, default="", help="Path to save KNN model to"
    )
    parser.add_argument(
        "--load-classifier-model",
        type=str,
        default="",
        help="Path to load classifier model from",
    )
    parser.add_argument(
        "--load-prfr-model", type=str, default="", help="Path to load PRFR model from"
    )
    parser.add_argument(
        "--load-knn-model", type=str, default="", help="Path to load KNN model from"
    )
    args = parser.parse_args()
    main()
