from argparse import ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bed_reader import to_bed

from simulations.wrapper import simulate_agg_risk_and_dis_outc
from simulations.risk_factors import simulate_risk_factors
from primer.preprocess import gaussianize
from primer.gwas import GWAS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--ne", type=int, default=50_000, help="Risk factors cohort size"
    )
    parser.add_argument(
        "--no", type=int, default=50_000, help="Outcome GWAS cohort size"
    )
    parser.add_argument(
        "--out_dir", type=Path, default="../data", help="Output dir"
    )
    parser.add_argument(
        "--num_rfs",
        dest="num_rfs",
        type=int,
        default=30,
        help="Number of risk factors",
    )
    parser.add_argument(
        "--num_qtls_rf",
        type=int,
        default=30,
        help="Number of QTLs for each risk factor",
    )
    parser.add_argument(
        "--num_shared_qtls_rf",
        type=int,
        default=5,
        help="Number of shared QTLs among risk factors",
    )
    parser.add_argument(
        "--var_geno",
        type=float,
        default=0.2,
        help="Variance explained cumulatively by SNPs on risk factors",
    )

    # Parameters for the simulating aggregated risk and outcome
    parser.add_argument(
        "--num_caus_rfs",
        dest="num_caus_rfs",
        type=int,
        default=10,
        help="Number of risk factors affecting outcome",
    )
    parser.add_argument(
        "--var_caus",
        dest="v_caus",
        type=float,
        default=0.8,
        help="Variance explained by risk factors on outcome",
    )
    parser.add_argument(
        "--var_hp",
        dest="v_hp",
        type=float,
        default=0.0,
        help="Variance explained by horizontal pleiotropy",
    )
    parser.add_argument(
        "--simulation_fn_name", dest="activation", type=str, default="elu"
    )
    parser.add_argument(
        "--simulation_fn_scale", dest="scale", type=float, default=np.power(10, 0.5)
    )
    parser.add_argument(
        "--simulation_fn_weight", dest="weight", type=float, default=0.0
    )

    return parser.parse_args()


def main(args):

    # set seed
    np.random.seed(args.seed)

    # total sample size
    Ntot = args.ne + args.no

    # sample genotypes
    nsnps = (
        args.num_rfs * (args.num_qtls_rf - args.num_shared_qtls_rf)
        + args.num_shared_qtls_rf
    )
    af = 0.02 + 0.18 * np.random.rand(nsnps)  # sample allele frequencies in [0.02, 0.2]
    G = np.random.binomial(2, af, size=(Ntot, af.shape[0]))
    dfG = pd.DataFrame(G, columns=[f"sid{i+1}" for i in range(G.shape[1])]).astype(
        np.float64
    )

    # sample covariates
    dfC = {}
    dfC["sex"] = np.random.randint(0, 2, size=Ntot)
    dfC["age"] = np.random.randint(40, 81, size=Ntot)
    dfC = pd.DataFrame(dfC).astype(np.float64)

    # simulate risk factors
    dfY = simulate_risk_factors(
        dfG,
        args.var_geno,
        args.num_rfs,
        args.num_shared_qtls_rf,
    )

    # simulate aggregate risk and outcome
    dfe, dfo = simulate_agg_risk_and_dis_outc(
        dfY,
        dfG,
        args.num_caus_rfs,
        args.v_caus,
        args.v_hp,
        args.activation,
        args.scale,
        args.weight,
    )

    # create splits
    # - osplit: outcome cohort, cohort where the outcome GWAS summary statistics is computed
    # - esplit: exposure cohort, cohort of healthy individuals with genetic data and risk factors
    esplit, osplit = train_test_split(
        np.arange(Ntot), test_size=args.no, random_state=args.seed, shuffle=False
    )

    # osplit
    dfo_o = dfo.loc[osplit]
    dfC_o = dfC.loc[osplit]
    dfG_o = dfG.loc[osplit]

    # Compute GWAS summary stats in osplit
    Fo = StandardScaler().fit_transform(dfC_o)
    Fo = np.concatenate([np.ones([Fo.shape[0], 1]), Fo], 1)
    Yo = gaussianize(dfo_o.values)
    gwas_o = GWAS(Yo, Fo)
    gwas_o.process(dfG_o.values)
    beta_o = gwas_o.getBetaSNP()
    beta_ste_o = gwas_o.getBetaSNPste()
    pv_o = gwas_o.getPv()
    df_ss = pd.DataFrame(
        np.concatenate([beta_o, beta_ste_o, pv_o], axis=1),
        index=dfG.columns,
        columns=["beta", "beta_ste", "pv"],
    )

    # esplit
    dfY_e = dfY.loc[esplit, :].reset_index(drop=True)
    dfG_e = dfG.loc[esplit, :].reset_index(drop=True)
    dfC_e = dfC.loc[esplit, :].reset_index(drop=True)
    dfe_e = dfe.loc[esplit, :].reset_index(drop=True)
    dfo_e = dfo.loc[esplit, :].reset_index(drop=True)

    # export files
    args.out_dir.mkdir(exist_ok=True, parents=True)
    dfY_e.to_parquet(args.out_dir / "risk_factors.parquet.gzip", compression="gzip")
    to_bed(filepath=args.out_dir / "genotypes.bed", val=dfG_e.values)
    dfC_e.to_parquet(args.out_dir / "covariates.parquet.gzip", compression="gzip")
    dfo_e.to_parquet(args.out_dir / "dis_outcome.parquet.gzip", compression="gzip")
    dfe_e.to_parquet(args.out_dir / "agg_risk_factor.parquet.gzip", compression="gzip")
    df_ss.to_csv(args.out_dir / "target_disease.sumstat", sep="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args)
