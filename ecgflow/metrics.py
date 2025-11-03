from sys import float_info
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import bz2
import dill
import numpy as np
from statsmodels.api import formula as smf
import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn
seaborn.set_style('whitegrid')

from scipy.stats import mannwhitneyu, norm, bootstrap
from sklearn import calibration as sk_calibration
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay) 
from skmisc import loess


def plot_roc(y_true, y_pred_prob, tpr, fpr, threshold, coloru='DodgerBlue',
             loader=None, id_list=[], ds_name='', calibration_nbins=12, 
             calibration_strategy='quantile', tag=None, outpath_base=None):
    """Plot discrimination (ROC curve) and calibration curves.
    
    Note: the `loader` must be instantiated with shuffle=False so that
    its id_list attribute (i.e., mrn) will have the same order as
    y_pred.

    """
    if loader:
        id_list = loader.dataset.id_list
        ds_name = loader.dataset.name
        save_pred = True
    elif len(id_list) > 0 and len(ds_name) > 0:
        save_pred = False  # pred's already saved
    else:
        raise ValueError('if `loader` is None, then both `id_list` and '
                         '`ds_name` must be given')
    if y_pred_prob.ndim == 2:
        y_pred = y_pred_prob.argmax(axis=1)
    else:
        y_pred = (y_pred_prob > 0.5).astype(np.int32)
    plt.figure(figsize=(6,6))
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(cm)  
    cmd.plot()
    plt.grid(0)
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'
        fn = f'{outpath_base}-cm.png'
        plt.savefig(fn)
        plt.close()

    plt.figure(figsize=(6,6))
    plt.plot([1,0], [0,1], '--', color='k')
    plt.plot(1 - fpr, tpr, '-', color=coloru)
    plt.gca().invert_xaxis()
    plt.xlabel('Specificity', fontsize=20)
    plt.ylabel('Sensitivity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    if outpath_base:
        fn = f'{outpath_base}-roc.png'
        plt.savefig(fn)
        plt.close()
    y_true = y_true[:, 0] if y_true.ndim == 2 else y_true
    y_pred_prob = y_pred_prob[:,1] if y_pred_prob.ndim == 2 else y_pred_prob
    cd = sk_calibration.CalibrationDisplay.from_predictions(
        y_true, y_pred_prob, n_bins=calibration_nbins,
        name=ds_name, strategy=calibration_strategy)
    if outpath_base:
        fn = f'{outpath_base}-calibration.png'
        cd.figure_.savefig(fn)
        plt.close()
    
    # save two dataframes
    df_roc = pandas.DataFrame(data=dict(zip(
        ['fpr', 'tpr', 'threshold'], [fpr, tpr, threshold])))
    fn = f'{outpath_base}-roc.csv'
    df_roc.to_csv(fn, index=False)
    if save_pred:
        df_pred = pandas.DataFrame(data=dict(zip(
            ['mrn', 'y_true', 'y_pred', 'y_pred_prob'], 
            [id_list, y_true, y_pred, y_pred_prob])))
        fn = f'{outpath_base}-pred.csv'
        df_pred.to_csv(fn, index=False)


@dataclass
class RocCurve(object):
    name: str
    tpr: np.array
    fpr: np.array
    threshold: np.array
    color: tuple=()
    auroc: float=-1.
        
    
@dataclass
class RegressionScatter(object):
    name: str
    y_pred: np.array
    y_true: np.array
    color: str
    slope: float=None
    intercept: float=None
    Rsq: float=None
    ylimU: float=None
    xlabelU: str='True'
    ylabelU: str='Predicted'

    
def load_roc(name, color, outpath_base, suffix='roc.csv'):
    """Return RocCurve instance after loading curve points from file.
    
    Parameters
    ----------
    name : str
        Unique name identifying the ROC curve
    color : str
        Color to use drawing the ROC curve
    outpath_base : str
        input path basename without filename extension
    suffix : str ['-roc.csv']
        suffix to be appended to `outpath_base`
    """
    fn = '-'.join([str(outpath_base), suffix])
    df = pandas.read_csv(fn)
    return RocCurve(name, df.tpr, df.fpr, df.threshold, color)
        

def load_regression_scatter(name, color, outpath_base, ylimU=None,
                            suffix='pred.csv',):
    fn = '-'.join([str(outpath_base), suffix])
    df = pandas.read_csv(fn)
    y_fit = smf.ols('y_pred ~ y_true', data=df).fit()
    slope, intc = y_fit.params['y_true'], y_fit.params['Intercept']
    rsq = y_fit.rsquared    
    return RegressionScatter(name, df.y_pred, df.y_true, color, 
                             slope, intc, rsq, ylimU)


def plot_roc_curves(roc_list, tag='', outpath_base=None):
    """Plot multiple ROC curves.
    
    Parameters
    ----------
    roc_list : list
        list containing RocCurve instances
    tag : str
        tag to identify output PNG file
    outpath_base : str
        output path basename without filename extension; 
        a suffix will be appended with '-{tag}-roc.png'
    """
    plt.figure(figsize=(6,6))
    plt.plot([1,0], [0,1], '--', color='black')
    for roc in roc_list:
        roc_label = f'{roc.name}'
        if roc.auroc > 0:
            roc_label = f'{roc_label} ({roc.auroc:0.3f})'
        plt.plot(1 - roc.fpr, roc.tpr, '-', color=roc.color, label=roc_label)
    plt.gca().invert_xaxis()
    plt.xlabel('Specificity', fontsize=20)
    plt.ylabel('Sensitivity', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='center right', fontsize=12)
    plt.tight_layout()
    if outpath_base:
        fn = f'{outpath_base}-{tag}-roc.png'
        print(fn)
        plt.savefig(fn)
        plt.close()
   
               
def combine_scatter_list(scatter_list, hue_name):
    """Return long-form DataFrame from given list of RegressionScatter instances. 
    """
    y_pred = np.array([it.y_pred for it in scatter_list]).ravel()
    y_true = np.array([it.y_true for it in scatter_list]).ravel()
    facet = np.array([len(it.y_pred)*[it.name] for it in scatter_list]).ravel()
    xlabelU, ylabelU = scatter_list[0].xlabelU, scatter_list[0].ylabelU
    cols = [ylabelU, xlabelU, hue_name]
    return pandas.DataFrame(data=dict(zip(cols, [y_pred, y_true, facet])),
                            columns=cols)
    

def plot_scatter_list(scatter_list, tag='', hue_name='Pretraining',
                      outpath_base=None):
    """Plot multiple scatter datasets on a single axis.
    
    Parameters
    ----------
    scatter_list : list
        list containing RegressionScatter instances
    tag : str
        tag to identify output PNG file
    outpath_base : str
        output path basename without filename extension; 
        a suffix will be appended with '-{tag}-scatter.png'
    """
    df = combine_scatter_list(scatter_list, hue_name)
    ylabelU, xlabelU = df.columns[:2]
    palette = [it.color for it in scatter_list]
    g = seaborn.relplot(data=df, x='True', y='Predicted', hue=hue_name,
                        kind='scatter', alpha=0.5, height=6, aspect=1,
                        palette=palette, facet_kws=dict(legend_out=False))
    g.facet_axis(0,0)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    imin = min(xlim[0], ylim[0])
    imax = max(xlim[1], ylim[1])
    plt.plot((imin, imax), (imin, imax), '--', color='k')  # line of identity
    for rs in scatter_list:
        xfit = np.arange(*xlim)
        yfit = rs.slope * xfit + rs.intercept
        labelfit = (f'{rs.name} Linear fit\nslope={rs.slope:0.3f}, '
                    f'intercept={rs.intercept:0.3f}')
        if rs.Rsq:
            labelfit += f'\nR$^2$={rs.Rsq:0.3f}'
        plt.plot(xfit, yfit, color=rs.color, label=labelfit)
        #plt.legend(loc='upper left', fontsize=12)
    g.set_axis_labels('True', 'Predicted', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    seaborn.move_legend(g, 'lower left', frameon=False)
    g.tight_layout()
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'        
        plt.savefig(f'{outpath_base}-scatter.png')
        plt.close()


def plot_multilabel_youden(df, outpath=None):
    n = df.shape[0]
    height = int(12. * n / 44)
    plt.figure(figsize=(height, 5))
    x = np.arange(n)
    plt.plot(x, df.auc, 'o-', color='DodgerBlue', lw=1, ms=3, label='AUROC')
    plt.plot(x, df.ICI, 'o-', color='DarkOrange', lw=1, ms=3, label='ICI')
    plt.legend(loc='center right')
    plt.xticks(x, df.index.values, rotation='vertical')
    plt.margins(0.05)
    plt.subplots_adjust(bottom=0.15)
    if outpath:
        plt.savefig(outpath)
        plt.close()


def _compute_metrics(sens, spec, npos, nneg):
    TP, TN = round(sens * npos, 0), round(spec * nneg, 0)
    FN, FP = round((1 - sens) * npos, 0), round((1 - spec) * nneg, 0)
    accuracy = (TP + TN) / float(npos + nneg)
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    return accuracy, ppv, npv
    
def youden_metrics(y_true, y_pred_prob, fpr, tpr, tag=None, 
                   optimal_threshold='both', outpath_base=None):
    """Return ROC metrics at two cut-points, at Youden max and minimium 
    distance to perfect discrimination (MDPD) (Sen/Spec 1,1)
    
    `optimal_threshold` : one of ['both', 'youden', 'mdpd']
    if None, both 'youden' and 'mdpd' threshold results will be returned
    
    Optionally, save the results to `outpath_base`-youden.csv
    """
    assert optimal_threshold in ('both', 'youden', 'mdpd')
    if hasattr(fpr, 'numpy'):
        fpr = fpr.numpy()
    if hasattr(tpr, 'numpy'):
        tpr = tpr.numpy()
    idxY = np.argmax((1 - fpr) + tpr - 1)
    spY, seY = (1 - fpr[idxY]), tpr[idxY]
    idxD = np.argmin((fpr[1:-1])**2 + (1 - tpr[1:-1])**2)
    spD, seD = (1 - fpr[1:-1][idxD]), tpr[1:-1][idxD]
    y0 = y_true[:,0] if y_true.ndim == 2 else y_true
    P, N = sum(y0==1), sum(y0==0)
    accY, ppvY, npvY = _compute_metrics(seY, spY, P, N)
    accD, ppvD, npvD = _compute_metrics(seD, spD, P, N)
    if y_pred_prob.ndim == 2:
        y_pred_prob = y_pred_prob[:,1]
    u0, u1 = y_pred_prob[y0==1], y_pred_prob[y0==0]
    mwu = mannwhitneyu(u0, u1)
    auc = mwu[0] / float(len(u0) * len(u1))
    tbl = dict(auc=[auc, auc],
               sensitivity=[seY, seD], specificity=[spY, spD],
               ppv=[ppvY, ppvD], npv=[npvY, npvD], accuracy=[accY, accD], 
               npos=[P, P], nneg=[N, N], total=[(P+N), (P+N)])
    df = pandas.DataFrame(data=tbl, index=['youden', 'mdpd'])
    thresh_tag = optimal_threshold
    if optimal_threshold == 'youden':
        res = df.loc['youden']
    elif optimal_threshold == 'mdpd':
        res = df.loc['mdpd']
    else:
        res = df
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'
        outpath = f'{outpath_base}-{thresh_tag}.csv'
        res.to_csv(outpath, index=False)
    return res
    

def multilabel_youden_metrics(y_true_arr, y_pred_arr, fpr_arr, tpr_arr, 
                              label_arr, tag=None, optimal_threshold='mdpd',
                              gen_roc_list=False, do_plot=True,
                              outpath_base=None):
    """Summarize multilabel binary classification results.
    """
    assert optimal_threshold in ('youden', 'mdpd')
    fn = f'{outpath_base}-pred-arr.npz'
    with open(fn, 'wb') as fp:
        np.savez_compressed(fp, y_true_arr=y_true_arr, y_pred_arr=y_pred_arr)
    resL, iciL = [], []
    if gen_roc_list:
        roc_list = []
        #cmap = plt.cm.get_cmap('hsv', len(label_arr))
        cmap = matplotlib.colormaps['hsv'].resampled(len(label_arr))
    for j, lbl in enumerate(label_arr):
        ytj, ypj = y_true_arr[:,j], y_pred_arr[:,j]
        fprj, tprj = fpr_arr[j], tpr_arr[j],
        dfj = youden_metrics(ytj, ypj, fprj, tprj,
                             optimal_threshold=optimal_threshold)
        resL.append(dfj)
        calj = calibration_metrics(ytj, ypj)
        iciL.append(calj['ICI'])
        if gen_roc_list:
            roc_list.append(RocCurve(lbl, tprj, fprj, np.array([0.]), cmap(j),
                                     dfj.loc['auc']))
    df = pandas.concat(resL, axis=1, ignore_index=True).T.set_index(label_arr)
    df['ICI'] = iciL
    thresh_tag = optimal_threshold
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'
        outpath = f'{outpath_base}-{thresh_tag}.csv'
        df.to_csv(outpath)
        if do_plot:
            plot_multilabel_youden(df, outpath.replace('.csv', '.png'))
        outpath = f'{outpath_base}-roc-list.pkl'
        if gen_roc_list:
            with open(outpath, 'wb') as fp:
                dill.dump(roc_list, fp)
            retval = df, roc_list
        else:
            retval = df
        # macro results
        macro_cols = ['auc', 'sensitivity', 'specificity', 'ppv', 'npv',
                      'accuracy', 'ICI']
        macro_data = dict(
            zip(
                df.columns, 
                [(df[it].mean(), df[it].std(), np.median(df[it])) 
                 for it in macro_cols]))
        dfmacro = pandas.DataFrame(macro_data, index=['mean', 'sd', 'median'])
        outpath = f'{outpath_base}-macro-roc.csv'
        dfmacro.to_csv(outpath)
        # 
    return retval


def load_roc_list(outpath_base):
    fn = f'{outpath_base}-roc-list.pkl'
    if not Path(fn).exists():
        raise ValueError(f'no such file {fn}')
    with open(fn, 'rb') as fp:
        roc_list = dill.load(fp)
    return roc_list
    
    
def roc_list_to_json(outpath_base):
    """Convert a list of RocCurve instances serialized to a pickle file
    to json.
    
    Assumes the RocCurve array members are actually torch.Tensor's.
    """
    roc_list = load_roc_list(outpath_base)
    rocL = []
    for rc in roc_list:
        rc.fpr = list(rc.fpr.numpy().astype(float))
        rc.tpr = list(rc.tpr.numpy().astype(float))
        rc.threshold = list(rc.threshold.astype(float))
        rc.auroc = float(rc.auroc)
        rocL.append(asdict(rc))
    fn = f'{outpath_base}-roc-list.jbz2'
    with bz2.BZ2File(fn, 'wb') as fp:
        bytes_out = json.dumps(rocL).encode()
        fp.write(bytes_out)
        print(fn)
        

def roc_list_from_json(outpath_base):
    fn = f'{outpath_base}-roc-list.jbz2'
    with bz2.BZ2File(fn, 'rb') as fp:
        rocin = json.loads(fp.read())
    rocL = []
    for d in rocin:
        rc = RocCurve(**d)
        rc.tpr = np.array(rc.tpr)
        rc.fpr = np.array(rc.fpr)
        rc.threshold = np.array(rc.threshold)
        rocL.append(rc)
    return rocL
    
    
def calibration_metrics(y_true, y_pred_prob, stderror=False,
                        tag=None, outpath_base=None):
    """Return dict of calibration metrics (ICI, E50, E90, Emax)
    
    cf. 
    Austin & Steyerberg (2019) paper on calibration metric $E_{max}$ 
    from `rms`
    
    Austin & Steyerberg (2014) "Graphical assessment of internal and
    external calibration of logistic regression models by using loess
    smoothers"
    """
    loess_calibrate = loess.loess(y_pred_prob, y_true)
    loess_calibrate.fit()
    y_pred_prob_calibrate = loess_calibrate.predict(
        y_pred_prob, stderror=stderror)
    abs_diff = np.abs(y_pred_prob_calibrate.values - y_pred_prob)
    ICI = np.mean(abs_diff)
    E50 = np.median(abs_diff)
    E90 = np.quantile(abs_diff, q=0.9)
    Emax = np.max(abs_diff)
    if outpath_base:
        data = dict(ICI=[ICI], E50=[E50], E90=[E90], Emax=[Emax])
        df = pandas.DataFrame(data=data)
        if tag:
            outpath_base = f'{outpath_base}-{tag}'
        outpath = f'{outpath_base}-calibration.csv'
        df.to_csv(outpath, index=False)
    return dict(ICI=ICI, E50=E50, E90=E90, Emax=Emax)


def bland_altman1(x, y, xlabelu, ylabelu, percent=True, 
                  ylimu=None, xlimu=None, alphau=0.5,
                  top=False, outpath=None, stats=(np.mean, np.std)):
    """Bland-Altman plot.

    Parameters
    ----------
    x, y : 1-d arrays
        continuous measurements to be compared (x is reference method)
    xlabelu, ylabelu : str
        labels for the plot
    percent : bool [True]
        rescale x and y values to percent
    ylimu, xlimu : float
        user-defined y- and x-axis limits (symmetric around 0)
    """
    seaborn.set_style('whitegrid')
    plt.figure()
    bax = stats[0](np.array([x, y]), axis=0)
    bay = y - x
    bax *= 100. if percent == True else 1.
    bay *= 100. if percent == True else 1.
    meandiff, loa = stats[0](bay), 1.96*stats[1](bay)
    baxmin, baxmax = bax.min(), bax.max()
    fbx = [baxmin, baxmax]
    yhi, ylo = meandiff + loa, meandiff - loa
    # print(bay.shape, loa.shape, yhi.shape)
    ylimu = 3. * max([abs(it) for it in [yhi, ylo]]) if not ylimu else ylimu
    ylimu *= 100. if percent == True else 1.
    plt.plot(bax, bay, 'o', color='DodgerBlue', alpha=alphau)
    plt.ylim(-ylimu, ylimu)
    plt.hlines(0, baxmin, baxmax, 'Gray', ':', lw=2)
    plt.hlines(meandiff, baxmin, baxmax, 'k', '--', lw=1)
    plt.fill_between(fbx, yhi, ylo, color='LightBlue', alpha=0.28)
    plt.hlines(yhi, baxmin, baxmax, 'DarkGray', '-', lw=0.75)
    plt.hlines(ylo, baxmin, baxmax, 'DarkGray', '-', lw=0.75)
    plt.xticks(fontsize=10), plt.yticks(fontsize=10)
    plt.xlabel(xlabelu, fontsize=12)
    plt.ylabel(ylabelu, fontsize=12)
    xlo, xhi = plt.xlim() if not xlimu else xlimu
    plt.xlim(xlo, xhi)
    yy0, yy1 = (0.93, 0.86) if top else (0.11, 0.04)
    meanLabel = 'Mean' if stats[0] == np.mean else 'Median'

    plt.text(0.98, yy0,
         '{1} Difference {0:3.2f}'.format(meandiff, meanLabel),
         ha='right', va='bottom', fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.98, yy1,
         'Limits of Agreement ({0:3.2f}, {1:3.2f})'
         .format(meandiff-loa, meandiff+loa),
         ha='right', va='bottom', fontsize=10, transform=plt.gca().transAxes)
    plt.tight_layout()
    if outpath != None:
        plt.savefig(outpath)
        plt.close()
    return meandiff, loa


def rsquared_ci(Rsq, n, k, conf=0.95):
    """cf. Cohen, J., Cohen, P., West, S. G., & Aiken,
    L. S. (2003). Applied multiple regression/correlation analysis for
    the behavioral sciences (3rd ed.). Mahwah: NJ: Erlbaum.
    """
    assert conf in (0.67, 0.80, 0.95, 0.99)
    se = (np.sqrt((4 * Rsq * ((1 - Rsq)**2) * ((n - k - 1)**2)) / 
                  ((n**2 - 1) * (n + 3))))
    pm = np.array([1, -1])
    if conf == 0.67:
        upper, lower = Rsq + pm * se
    elif conf == 0.8:
        upper, lower = Rsq + pm * 1.3 * se
    elif conf == 0.95:
        upper, lower = Rsq + pm * 2 * se
    elif conf == 0.99:
        upper = Rsq + pm * 2.6 * se
    return se, (lower, upper)


def rsquared_test(Rsq1, Rsq2, se1, se2, n1, n2, pooled=False):
    if pooled == False:
        se_diff = np.sqrt(se1**2 + se2**2)
    else:
        se_diff = (np.sqrt(((se1**2) * (n1 - 1) + (se2**2) * (n2 - 1)) / 
                           (n1 + n2 - 2)))
    Rsq_diff = Rsq1 - Rsq2
    z = Rsq_diff / se_diff
    p = 2 * (1 - norm.cdf(z))
    return z, p


def compute_auroc(y_pred, y_true):
    u0, u1 = y_pred[y_true==0], y_pred[y_true==1]
    mwu = mannwhitneyu(u0, u1)
    return mwu[0] / float(len(u0) * len(u1))
    

def auroc_ci(y_pred, y_true, paired=True, conf=0.95, n_resamples=10000,
             seed=None, rng=None):
    auroc = compute_auroc(y_pred, y_true)
    rng = np.random.default_rng(seed) if not rng else rng
    res = bootstrap((y_pred, y_true), compute_auroc,
                     n_resamples=n_resamples, paired=paired,
                     confidence_level=conf, random_state=rng)
    return auroc, res


def multilabel_macro_auroc_ci(y_pred_arr, y_true_arr, label_arr, 
                              paired=True, conf=0.95, n_resamples=10000,
                              seed=None):
    """Compute macro AUROC with bootstrap SE and CI
    """
    aurocL, seL, ciL = [], [], []
    rng = np.random.default_rng(seed)
    for j, lbl in enumerate(label_arr):
        ytj, ypj = y_true_arr[:,j], y_pred_arr[:,j]
        auroc, bs = auroc_ci(ypj, ytj, paired=paired, conf=conf, 
                             n_resamples=n_resamples, rng=rng)
        aurocL.append(auroc)
        seL.append(bs.standard_error)
        ciL.append(bs.confidence_interval)
    seL = np.array(seL)
    ciL = np.array(ciL)
    ciL_lower, ciL_upper = ciL[:,0], ciL[:,1]
    return dict(macro_auroc=np.mean(aurocL), 
                se=np.sqrt(np.mean(seL**2)),
                ci=(np.mean(ciL_lower), np.mean(ciL_upper)))


def compute_mae(y_pred, y_true):
    return np.mean(np.absolute(y_pred - y_true))


def mae_ci(y_pred, y_true, paired=True, conf=0.95, n_resamples=10000,
           seed=None):
    """Bootstrap CI for Mean Absolute Error
    """
    mae = compute_mae(y_pred, y_true)
    rng = np.random.default_rng(seed)
    res = bootstrap((y_pred, y_true), compute_mae,
                     n_resamples=n_resamples, paired=paired,
                     confidence_level=conf, random_state=rng)
    return mae, res


def mae_paired_test(y_pred1, y_pred2, y_true, conf=0.95, n_resamples=10000,
                    seed=None):
    """Paired test of two MAE's with bootstrap.
    """
    assert len(y_pred1) == len(y_pred2)
    y_pred1, y_pred2 = np.asarray(y_pred1), np.asarray(y_pred2)
    y_true = np.asarray(y_true)
    mae1 = np.mean(np.absolute(y_pred1 - y_true))
    mae2 = np.mean(np.absolute(y_pred2 - y_true))
    idx = np.arange(len(y_pred1)).astype(np.int32)
    rng = np.random.default_rng(seed)
    res = bootstrap((idx,),
                    lambda idx:
                        (np.mean(np.absolute(y_pred1[idx] - y_true[idx])) -
                         np.mean(np.absolute(y_pred2[idx] - y_true[idx]))),
                    n_resamples=n_resamples, paired=True,
                    confidence_level=conf, random_state=rng
                    )
    d = res.bootstrap_distribution
    pval = max(1 - norm.cdf(0., loc=d.mean(), scale=d.std()),
               float_info.min)
    return dict(mae1=mae1, mae2=mae2, diff=(mae1 - mae2), pval=pval, bs=res)
    

def regression_metrics(y_pred, y_true, loader=None, id_list=[], scaler=None, 
                       tag=None, outpath_base=None, verbose=False):
    if scaler:
        y_pred = scaler.unscale(y_pred)
        y_true = scaler.unscale(y_true)
    if loader:
        id_list = loader.dataset.id_list
        save_pred = True
    elif len(id_list) > 0:
        save_pred = False  # pred's are already saved with `id_list`
    else:
        raise ValueError('if `loader` is None, `id_list` must be given')
    y_squared_error = (y_pred - y_true)**2
    y_rmse = np.sqrt(np.mean(y_squared_error))
    y_sdse = np.sqrt(np.var(y_squared_error))
    y_abs_error = abs(y_pred - y_true)
    y_mae = np.mean(y_abs_error)
    y_sdae = np.std(y_abs_error)
    y_true = y_true.squeeze(1) if y_true.ndim > 1 else y_true
    y_pred = y_pred.squeeze(1) if y_pred.ndim > 1 else y_pred
    y_df = pandas.DataFrame(dict(zip(
        ['mrn', 'y_true', 'y_pred'], [id_list, y_true, y_pred])))
    y_fit = smf.ols('y_pred ~ y_true', data=y_df).fit()
    if verbose:
        print(y_fit.summary2())
    slope, intc = y_fit.params['y_true'], y_fit.params['Intercept']
    rsq = y_fit.rsquared
    m_dict = dict(rmse=[y_rmse], rsdse=[y_sdse], 
                  mae=[y_mae], sdae=[y_sdae], 
                  slope=[slope], intercept=[intc], 
                  Rsq=[rsq])
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'
        if save_pred:
            pred_fn = f'{outpath_base}-pred.csv'
            y_df.to_csv(pred_fn, index=False)
        df = pandas.DataFrame(m_dict)
        outpath = f'{outpath_base}-regression-metrics.csv'
        df.to_csv(outpath, index=False)
    return y_pred, y_true, m_dict

    
def plot_regression(y_pred, y_true, slope=None, intercept=None, Rsq=None,
                    xlabelU='', ylabelU='', ylimu=None, show_percent=False,
                    tag=None, outpath_base=None):
    if show_percent:
        y_true *= 100.
        y_pred *= 100.
        intercept *= 100.
    g = seaborn.relplot(x=y_true, y=y_pred, kind='scatter', alpha=0.5)
    g.facet_axis(0,0)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    imin = min(xlim[0], ylim[0])
    imax = max(xlim[1], ylim[1])
    plt.plot((imin, imax), (imin, imax), '--', color='k')  # line of identity
    if slope and intercept:
        xfit = np.arange(*xlim)
        yfit = slope * xfit + intercept
        labelfit = f'Linear fit\nslope={slope:0.3f}, intercept={intercept:0.3f}'
        if Rsq:
            labelfit += f'\nR$^2$={Rsq:0.3f}'
        plt.plot(xfit, yfit, color='DarkOrange', label=labelfit)
        plt.legend(loc='upper left', fontsize=12)
    plt.xlabel(xlabelU)
    plt.ylabel(ylabelU)
    if outpath_base:
        if tag:
            outpath_base = f'{outpath_base}-{tag}'        
        plt.savefig(f'{outpath_base}-scatter.png')
        plt.close()
    plt.figure()
    mean_diff, loa = bland_altman1(y_true, y_pred, alphau=0.3,
                                   xlabelu='Mean (Predicted, Actual)', 
                                   ylabelu='Diff (Predicted, Actual)',
                                   percent=False, ylimu=ylimu)
    if outpath_base:
        plt.savefig(f'{outpath_base}-bland-altman.png')
        plt.close()
    return g, mean_diff, loa
