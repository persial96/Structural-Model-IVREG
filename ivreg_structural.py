import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, List
from patsy import dmatrices
from scipy import stats

# Helpers
def _rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X))

def _ols_fit(X: np.ndarray, y: np.ndarray):
    # Stable OLS via least squares; works even if X is rank-deficient.
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    df_resid = n - _rank(X)
    s2 = float((resid.T @ resid) / df_resid)
    return beta, resid, fitted, s2, df_resid

def _f_test_nested(X_r, y, X_ur):
    # Classic nested F-test comparing restricted (X_r) and unrestricted (X_ur)
    _, _, _, s2_ur, df2_ur = _ols_fit(X_ur, y)
    _, _, _, s2_r, df2_r  = _ols_fit(X_r,  y)
    n = X_ur.shape[0]
    p_ur = _rank(X_ur)
    p_r  = _rank(X_r)
    m = p_ur - p_r
    RSS_ur = s2_ur * (n - p_ur)
    RSS_r  = s2_r  * (n - p_r)
    F = float(((RSS_r - RSS_ur)/m) / (RSS_ur/(n - p_ur)))
    df1, df2 = m, (n - p_ur)
    pval = 1 - stats.f.cdf(F, df1, df2)
    return F, df1, df2, pval

def _nR2_test(Z, e, df):
    # Sargan J = n*R^2 from regression of IV residuals on all instruments Z
    _, resid, _, _, _ = _ols_fit(Z, e)
    n = Z.shape[0]
    ybar = float(e.mean())
    TSS = float(((e - ybar)**2).sum())
    RSS = float((resid**2).sum())
    R2 = 0.0 if TSS == 0 else 1 - RSS/TSS
    J = n * R2
    p = 1 - stats.chi2.cdf(J, df)
    return float(J), float(p)

def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Keep first occurrence of any duplicated column name (patsy can repeat)
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()

# IVRES
@dataclass
class IVResults:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    residuals: pd.Series
    fittedvalues: pd.Series
    df_resid: int
    df_model: int
    nobs: int
    r2: float
    r2_adj: float
    cov: np.ndarray
    first_stage_F: Optional[Tuple[float,int,int,float]]
    wu_hausman: Optional[Tuple[float,int,int,float]]
    sargan: Optional[Tuple[float,int,float]]
    call: str

    def summary(self) -> str:
        head = (
            "2SLS regression via IV2SLS\n"
            f"Call: {self.call}\n"
            f"Observations: {self.nobs}    Residual DF: {self.df_resid}    Model DF: {self.df_model}\n\n"
            "Coefficients:\n"
        )
        tbl = pd.DataFrame({
            "coef": self.params,
            "std err": self.bse,
            "t": self.tvalues,
            "P>|t|": self.pvalues
        })
        meta = (
            f"\nR-squared: {self.r2:.4f}    Adjusted R-squared: {self.r2_adj:.4f}\n"
        )
        diag = []
        if self.first_stage_F is not None:
            F, df1, df2, p = self.first_stage_F
            diag.append(f"Weak instruments test: F({df1}, {df2}) = {F:.3f}, p-value = {p:.4g}")
        if self.wu_hausman is not None:
            Fh, d1, d2, ph = self.wu_hausman
            diag.append(f"Wu–Hausman endogeneity test: F({d1}, {d2}) = {Fh:.3f}, p-value = {ph:.4g}")
        if self.sargan is not None:
            J, dfJ, pJ = self.sargan
            diag.append(f"Sargan overidentification test: χ²({dfJ}) = {J:.3f}, p-value = {pJ:.4g}")
        return head + tbl.to_string() + meta + ("\n" + "\n".join(diag) if diag else "")

# Define estimator (two-stage)
class IV2SLS:
    """
    Two-Stage Least Squares with ivreg-like formula:
    'y ~ x1 + x2 + ... | z1 + z2 + ...'
    All exogenous regressors must also appear among the instruments.
    """
    def __init__(self, formula: str, data: pd.DataFrame):
        if "|" not in formula:
            raise ValueError("Formula must be 'y ~ regressors | instruments'")
        self.formula = formula
        self.data = data.copy()
        self._X_names: List[str] = []
        self._Z_names: List[str] = []

    def _parse(self):
        left, right = self.formula.split("~", 1)
        rhs_reg, rhs_iv = [s.strip() for s in right.split("|", 1)]
        yX = dmatrices(f"{left.strip()} ~ {rhs_reg}", self.data, return_type="dataframe")
        yZ = dmatrices(f"{left.strip()} ~ {rhs_iv}",  self.data, return_type="dataframe")

        y = yX[0].iloc[:,0].to_frame()
        X = _dedup_cols(yX[1])
        Z = _dedup_cols(yZ[1])

        # drop any rows with NA across used columns
        used = list(pd.Index(X.columns).union(Z.columns)) + [y.columns[0]]
        df_all = pd.concat([y, X, Z], axis=1).dropna(subset=used)
        y = df_all.iloc[:, [0]]
        X = df_all.loc[:, X.columns]
        Z = df_all.loc[:, Z.columns]

        self._X_names = list(X.columns)
        self._Z_names = list(Z.columns)
        return y.values.ravel().astype(float), X.values.astype(float), Z.values.astype(float)

    def fit(self, diagnostics: bool = True) -> IVResults:
        y, X, Z = self._parse()
        n, p = X.shape
        L = Z.shape[1]

        # Stage 1 projection: Xhat = Z (Z'Z)^+ Z' X
        A = np.linalg.lstsq(Z, X, rcond=None)[0]          # L×p
        Xhat = Z @ A                                       # n×p

        # Stage 2 OLS: y ~ Xhat
        beta, resid, fitted, s2, df_resid = _ols_fit(Xhat, y)

        # Homoskedastic covariance (use pseudoinverse to avoid LinAlgError)
        XtX = Xhat.T @ Xhat
        cov = s2 * np.linalg.pinv(XtX)
        se = np.sqrt(np.diag(cov))
        tvals = beta / se
        pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df_resid))

        # R^2 (relative to y)
        ybar = float(y.mean())
        TSS = float(((y - ybar)**2).sum())
        RSS = float((resid**2).sum())
        R2 = 0.0 if TSS == 0 else 1 - RSS/TSS
        R2_adj = 1 - (1 - R2) * (n - 1) / df_resid

        # ---- diagnostics ----
        first_stage_F = wu_hausman = sargan = None
        if diagnostics:
            X_names = self._X_names
            Z_names = self._Z_names

            # Exogenous regressors W = intersection(X,Z). Endogenous = X \ Z.
            W_idx = [i for i, nm in enumerate(X_names) if nm in Z_names]
            EN_idx = [i for i, nm in enumerate(X_names) if nm not in Z_names]
            W = X[:, W_idx] if W_idx else np.empty((n,0))
            EN = X[:, EN_idx] if EN_idx else np.empty((n,0))

            # Excluded instruments Q = Z \ X
            Q_cols = [j for j, nm in enumerate(Z_names) if nm not in X_names]
            Q = Z[:, Q_cols] if Q_cols else np.empty((n,0))

            # Weak-IV F: for each endogenous regressor, test whether Q adds explanatory power beyond W.
            if EN.shape[1] > 0 and Q.shape[1] > 0:
                Fs = []
                Xr = W if W.shape[1] > 0 else np.ones((n,1))
                Xur = np.column_stack([W, Q]) if W.shape[1] > 0 else Q
                for j in range(EN.shape[1]):
                    xj = EN[:, j]
                    Fs.append(_f_test_nested(Xr, xj, Xur))
                # report the minimum F across endogenous regressors (conservative)
                first_stage_F = min(Fs, key=lambda t: t[0])

            # Wu–Hausman via residual-inclusion:
            if EN.shape[1] > 0:
                R_stack = []
                for j in range(EN.shape[1]):
                    xj = EN[:, j]
                    X1 = np.column_stack([W, Z]) if W.shape[1] > 0 else Z
                    _, rj, _, _, _ = _ols_fit(X1, xj)
                    R_stack.append(rj.reshape(-1,1))
                Rmat = np.column_stack(R_stack)
                X_aug = np.column_stack([X, Rmat])   # use original X, not Xhat
                Fh, d1, d2, ph = _f_test_nested(X, y, X_aug)
                wu_hausman = (Fh, d1, d2, ph)

            # Sargan over-ID: only if overidentified (L > p)
            if L > p:
                e = resid.reshape(-1,1)
                dfJ = L - p
                J, pJ = _nR2_test(Z, e, dfJ)
                sargan = (J, int(dfJ), pJ)

        df_model = _rank(X) - 1 if "Intercept" in self._X_names else _rank(X)
        res = IVResults(
            params=pd.Series(beta, index=self._X_names),
            bse=pd.Series(se,   index=self._X_names),
            tvalues=pd.Series(tvals, index=self._X_names),
            pvalues=pd.Series(pvals, index=self._X_names),
            residuals=pd.Series(resid, name="residuals"),
            fittedvalues=pd.Series(fitted, name="fitted"),
            df_resid=int(df_resid),
            df_model=int(df_model),
            nobs=int(n),
            r2=R2, r2_adj=R2_adj,
            cov=cov,
            first_stage_F=first_stage_F,
            wu_hausman=wu_hausman,
            sargan=sargan,
            call=self.formula
        )
        return res

#### Simulation ---------------------------------------------------------------
def simulate_panel_data(n_groups=20, n_periods=10, seed=123, iv_strength=0.8):
    """
    Balanced panel with endogenous price and excluded instrument.
    Returns a DataFrame with columns:
      group, time, log_Q, log_P, log_Coupon, log_BidVolumeTotal, log_inf, log_ret_SMI
    """
    rng = np.random.default_rng(seed)
    N = n_groups * n_periods
    g = np.repeat(np.arange(n_groups), n_periods)
    t = np.tile(np.arange(n_periods), n_groups)

    # exogenous drivers
    coupon = rng.normal(0, 0.2, N)
    infl   = rng.normal(0, 0.2, N)
    ret    = rng.normal(0, 0.3, N)
    bidvol = rng.normal(0, 1.0, N)  # excluded instrument

    # structural shocks with correlation to make price endogenous
    u = rng.normal(0, 1.0, N)
    v = 0.6*u + rng.normal(0, 1.0, N)

    # first stage: price depends on instrument + exogenous
    log_P = 0.4*coupon + 0.2*infl + iv_strength*0.8*bidvol + v

    # demand (true elasticities around -1)
    log_Q = 0.5 - 1.1*log_P + 0.3*coupon - 0.2*infl + 0.1*ret + u

    df = pd.DataFrame({
        "group": g, "time": t,
        "log_Q": log_Q,
        "log_P": log_P,
        "log_Coupon": coupon,
        "log_BidVolumeTotal": bidvol,
        "log_inf": infl,
        "log_ret_SMI": ret
    })
    return df

def plot_diagnostics(ivres: IVResults, X: pd.DataFrame=None, Z: pd.DataFrame=None):
    """
    Minimal plotting helper (uses matplotlib). Call with ivres from .fit().
    Returns a dict of matplotlib figures.
    """
    import matplotlib.pyplot as plt
    figs = {}

    # Residual histogram
    fig1 = plt.figure()
    plt.hist(ivres.residuals, bins=30)
    plt.title("Residuals histogram")
    figs["resid_hist"] = fig1

    # QQ plot
    from scipy.stats import probplot
    fig2 = plt.figure()
    probplot(ivres.residuals, dist="norm", plot=plt)
    plt.title("Normal Q–Q")
    figs["qq"] = fig2

    # Fitted vs observed
    fig3 = plt.figure()
    plt.scatter(ivres.fittedvalues, ivres.fittedvalues + ivres.residuals, s=10)
    plt.xlabel("Fitted"); plt.ylabel("Observed"); plt.title("Fitted vs Observed")
    figs["fit_vs_obs"] = fig3

    # If caller provides X and Z, try an instrument vs price plot
    if X is not None and Z is not None:
        # pick first endogenous x and first excluded instrument if available
        x_names = list(X.columns)
        z_names = list(Z.columns)
        endog = [nm for nm in x_names if nm not in z_names and nm != "Intercept"]
        excl  = [nm for nm in z_names if nm not in x_names and nm != "Intercept"]
        if endog and excl:
            fig4 = plt.figure()
            plt.scatter(Z[excl[0]], X[endog[0]], s=10)
            plt.xlabel(excl[0]); plt.ylabel(endog[0]); plt.title("Instrument vs endogenous regressor")
            figs["inst_vs_endog"] = fig4

    return figs
