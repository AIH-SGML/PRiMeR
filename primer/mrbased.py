from typing import List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st


class UVMRbased:
    def __init__(self, pv_iv: float = 5e-8, min_ivs: int = 5, fwer_exp: float = 0.05):
        """
        Initialize the UVMRbased instance with parameters for the analysis.

        :param pv_iv: P-value inclusion threshold for selecting instruments.
        :param min_ivs: Minimum number of instruments required to perform regression.
        :param fwer_exp: Family-wise error rate correction of significant regression slope.
        """
        self.pv_iv = pv_iv
        self.min_ivs = min_ivs
        self.fwer_exp = fwer_exp
        self.dfres = None

    def train(
        self,
        beta_e: np.ndarray,
        pv_e: np.ndarray,
        beta_o: np.ndarray,
        ste_o: np.ndarray,
        exp_keys: List[str],
    ):
        """
        Train the UVMR model using provided genetic association data.

        :param beta_e: Effect estimates of the genetic variants on the risk factors.
        :param pv_e: P-values associated with the genetic variants' effects on risk factors.
        :param beta_o: Effect estimates of the genetic variants on the outcome.
        :param ste_o: Standard errors of the genetic variants' effects on the outcome.
        :param exp_keys: List of names representing the risk factors.
        :return: A DataFrame with results for each exposure trait including estimated effect, standard error, and p-value.
        """

        # Define quantities for UVMR
        y = beta_o  # Outcome effects
        w = 1 / ste_o**2  # Weights for WLS regression, inverse of variance
        X = beta_e  # Exposure effects
        pv = pv_e  # Exposure p-values

        # Run UVMR analysis for each exposure
        dfres = []
        for ik, key in enumerate(exp_keys):
            _dfres = {}
            _dfres["trait"] = key
            Ikeep = (
                pv[:, ik] < self.pv_iv
            )  # Filter instruments based on p-value threshold
            if (
                Ikeep.sum() >= self.min_ivs
            ):  # Ensure sufficient instruments are available
                _x = X[Ikeep, [ik]]  # Selected exposure effects
                _y = y[Ikeep]  # Selected outcome effects
                _w = w[Ikeep]  # Weights for the selected instruments
                mod = sm.WLS(
                    _y, _x, weights=_w
                ).fit()  # Perform weighted least squares regression
                _dfres["bm"] = mod.params[0]  # Regression coefficient
                _dfres["bs"] = mod.bse[0]  # Standard error of regression coefficient
                _dfres["pv"] = st.chi2(1).sf(
                    (_dfres["bm"] / _dfres["bs"]) ** 2
                )  # P-value from chi-squared test
            else:
                _dfres["bm"] = 0.0  # Default values if not enough instruments
                _dfres["bs"] = 0.0
                _dfres["pv"] = 1.0
            dfres.append(_dfres)
        self.dfres = pd.DataFrame(dfres)

        return self.dfres.sort_values("pv")

    def predict(self, E: np.ndarray):
        """
        Predict outcomes using the trained model and risk factors.

        :param E: Risk factors matrix used to predict outcomes.
        :return: Predicted outcomes as a numpy array.
        """
        assert self.dfres is not None, "run train first!"  # Ensure the model is trained
        pv_exp = (
            self.fwer_exp / self.dfres.shape[0]
        )  # Adjust p-value threshold for multiple comparisons
        I = (self.dfres["pv"] < pv_exp).values  # Identify significant exposures
        if I.sum() == 0:
            y_pred = np.zeros(
                [E.shape[0], 1]
            )  # Return zero array if no significant exposures
        else:
            y_pred = E[:, I].dot(self.dfres["bm"].values[I])[
                :, None
            ]  # Compute predictions

        return y_pred


class MVMRbased:
    def __init__(self):
        self.dfres = None

    def train(
        self,
        beta_e: np.ndarray,
        beta_o: np.ndarray,
        ste_o: np.ndarray,
        exp_keys: List[str],
    ):
        """
        Fit the multivariable Mendelian randomization model using input genetic associations.

        :param beta_e: Coefficients for each genetic instrument per exposure.
        :param beta_o: Outcome coefficients for each genetic instrument.
        :param ste_o: Standard errors for each genetic instrument's outcome coefficient.
        :param exp_keys: Names of exposures.
        """
        y = beta_o
        w = 1 / ste_o**2
        X = beta_e

        # Fit MVMR using Weighted Least Squares
        mod = sm.WLS(y, X, weights=w).fit()

        # Store results in DataFrame
        dfres = pd.DataFrame(
            {
                "trait": exp_keys,
                "bm": mod.params,
                "bs": mod.bse,
                "pv": st.chi2(1).sf((mod.params / mod.bse) ** 2),
            }
        )

        self.dfres = dfres

        return self.dfres.sort_values("pv")

    def predict(self, E: np.ndarray):
        """
        Predict outcomes based on the fitted MVMR model and exposure data.

        :param E: Risk factor data matrix to predict outcomes.
        :return: Predicted outcomes as a numpy array.
        """
        assert (
            self.dfres is not None
        ), "Model must be trained using train() before predictions can be made."

        # Compute predictions
        y_pred = E.dot(self.dfres["bm"].values)[:, None]
        
        return y_pred
