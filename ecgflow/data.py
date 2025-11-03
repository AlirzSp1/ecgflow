import ast
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from .data_wfdb import init_wfdb

PTBXL_DATABASE_FILE = 'ptbxl_database.csv'
PTBXL_SCP_FILE = 'scp_statements.csv'
PTBXL_LABEL_AUX_STATS = 'ptbxl-label-aux-stats.csv'

# These ECG scaling constants are in mV over all leads
MIMICIV_12CHANNEL_MEAN = dict(rest=0.005976)
MIMICIV_12CHANNEL_SD = dict(rest=0.195642)
MIMICIV_8CHANNEL_MEAN = dict(rest=0.009892)
MIMICIV_8CHANNEL_SD = dict(rest=0.216624)
PTBXL_12CHANNEL_MEAN = dict(rest=-0.00078524655)
PTBXL_12CHANNEL_SD = dict(rest= 0.23621394)
PTBXL_8CHANNEL_MEAN = dict(rest=-0.0012090115)
PTBXL_8CHANNEL_SD = dict(rest=0.26843855)


def load_scale_constants(fields, file_in):
    dfin = pd.read_csv(file_in, index_col=0)
    df_index = dfin.index
    y_loc, y_scale = [], []
    for y in fields:
        if y in df_index:
            y_loc.append(dfin.loc[y].Mean)
            y_scale.append(dfin.loc[y].SD)
        else:
            raise ValueError(f'field {y} not found in {file_in}')
    if len(fields) == 1:
        y_loc, y_scale = y_loc[0], y_scale[0]
    return y_loc, y_scale


def read_ptbxl_database(data_dir):
    dfp = pd.read_csv(data_dir/PTBXL_DATABASE_FILE, index_col='ecg_id')
    dfp['scp_codes'] = dfp.scp_codes.apply(lambda x: ast.literal_eval(x))
    dfp['bmi'] = dfp.weight / (dfp.height / 100.)**2
    return dfp


def read_ptbxl_scp_codes(data_dir):
    return pd.read_csv(data_dir/PTBXL_SCP_FILE, index_col=0)


def compute_label_aggregations(df, ctype, dfscp):
    """Return selected PTB-XL labels based on prediction task `ctype`
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame instance containing table in PTB-XL database file
    ctype : str
        prediction task, one of ['diagnostic', 'subdiagnostic', 
        'superdiagnostic', 'form', 'rhythm', 'all']
    dfscp : pandas.DataFrame
        DataFrame instance containing table in scp_statements.csv file
    
    Return
    ------
    input DataFrame `df` with two new columns named `ctype` and `ctype`_len
    
    This function was borrowed from the utils module of:
    http://github.com/helme/ecg_ptbxl_benchmarking
    """
    aggregation_df = dfscp
    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all_scp':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))
        df['all_scp_len'] = df.scp_codes.apply(lambda x: len(x))

    return df


def select_data_for_ptbxl_task(df, ctype, min_samples=0):
    """Return selected PTB-XL ECG rows based on prediction task `ctype`
    and labels converted to one-hot encoding.
    
    Parameters
    ----------
    file_list_in : list
        input list of raw ECG file paths from PTB-XL database
    df : pandas.DataFrame
        DataFrame instance containing table in PTB-XL database file
        after calling `compute_label_aggregations()`; must contain
        a column `ctype`_len
    ctype : str
        prediction task, one of ['diagnostic', 'subdiagnostic', 
        'superdiagnostic', 'form', 'rhythm', 'all']
    min_samples : int [0]
        cases in `df` with <= `min_samples` labels for given `ctype` 
        will be excluded
    
    Return
    ------
    2-tuple (df, label) 
    `df` : input DataFrame `df` with label column `ctype` updated
    `label` : one-hot encoded label for task `ctype`
    
    This function was borrowed from the utils module of:
    http://github.com/helme/ecg_ptbxl_benchmarking
    """   
    mlb = MultiLabelBinarizer()
    if ctype == 'diagnostic':
        dfout = df[df.diagnostic_len > 0]
        mlb.fit(dfout.diagnostic.values)
        y = mlb.transform(dfout.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(
            np.concatenate(df.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        df.subdiagnostic = df.subdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        dfout = df[df.subdiagnostic_len > 0]
        mlb.fit(dfout.subdiagnostic.values)
        y = mlb.transform(dfout.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(
            np.concatenate(df.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        df.superdiagnostic = df.superdiagnostic.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
        dfout = df[df.superdiagnostic_len > 0]
        mlb.fit(dfout.superdiagnostic.values)
        y = mlb.transform(dfout.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(
            np.concatenate(df.form.values)).value_counts()
        counts = counts[counts > min_samples]
        df.form = df.form.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        df['form_len'] = df.form.apply(lambda x: len(x))
        # select
        dfout = df[df.form_len > 0]
        mlb.fit(dfout.form.values)
        y = mlb.transform(dfout.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(
            np.concatenate(df.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        df.rhythm = df.rhythm.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
        # select
        dfout = df[df.rhythm_len > 0]
        mlb.fit(dfout.rhythm.values)
        y = mlb.transform(dfout.rhythm.values)
    elif ctype == 'all_scp':
        # filter 
        counts = pd.Series(
            np.concatenate(df.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        df.all_scp = df.all_scp.apply(
            lambda x: list(set(x).intersection(set(counts.index.values))))
        df['all_scp_len'] = df.all_scp.apply(lambda x: len(x))
        # select
        dfout = df[df.all_scp_len > 0]
        mlb.fit(dfout.all_scp.values)
        y = mlb.transform(dfout.all_scp.values)
    else:
        raise ValueError(f'unsupported `ctype` {ctype}')
    return dfout, y, mlb.classes_


def select_data_no_nan(df, ykey):
    """Return numpy bool array of complete rows in `df` (nan's excluded)
    """
    idx = np.ones(df.shape[0]).astype(bool)
    for yk in ykey:
        idx &= ~df.loc[:, yk].isna()
        if yk == 'age':
            idx &= ~(df.loc[:, yk] == 300).values
    return idx


class EcgBaseWfdbData(object):
    """ECG data in wfdb format.
    
    Parameters
    ----------
    data_dir : str
        path to top-level directory containing data
    use_split : str ['train']
        split of data to return, one of 'train', 'val', 'test'
    split_ratio : 3-tuple [0.975, 0.025, 0.]
        split fractions for (train, val, test)
    **kwargs : 
        optional kwargs passed to `get_records()`
    """
    def __init__(self, data_dir=None, use_split='train', 
                 split_ratio=(0.975, 0.025, 0.), 
                 ecg_filter=False, scale_y=False, **kwargs):
        self.data_dir = data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError('`data_dir` not found')
        self.use_split = use_split
        self.split_ratio = split_ratio
        id_list, record_list = self.get_records(**kwargs)
        id_list, record_list = self.get_split(id_list, record_list)
        self.id_list = id_list
        self.record_list = record_list
        self.trim_channels = kwargs.get('trim_channels', False)
        self.ecg_filter = ecg_filter
        self.scale_y = scale_y
        self.transform = kwargs.get('transform', {})
        
    def get_records(self, **kwargs):
        raise NotImplementedError('Must override in subclass')

    def get_split(self, id_list, record_list):
        use_split, split_ratio = self.use_split, self.split_ratio
        ntotal = len(record_list)
        nval, ntest = (np.floor(np.array(split_ratio[1:]) * ntotal)
                       .astype(np.int32))
        ntrain = ntotal - nval - ntest
        if use_split == 'train':
            slc = slice(0, ntrain)
        elif use_split == 'val':
            slc = slice(ntrain, ntrain + nval)
        else:
            slc = slice(ntrain + nval, ntrain + nval + ntest)
        return id_list[slc], record_list[slc]
    

class EcgMimicIvData(EcgBaseWfdbData):
    """MIMIC-IV ECG data
    
    No test split is necessary since this unlabeled data will only be used for
    self-supervised pretraining.
    
    (Total):Train/Val/Test split (97.5/2.5/0)
    800,035:780,034/20,001/0
    """
    def __init__(self, data_dir=None, **kwargs):
        super().__init__(data_dir, **kwargs)
        init_wfdb(data_dir)
        self.xkey_name = ['rest']
        if self.trim_channels:
            self.mean = MIMICIV_8CHANNEL_MEAN
            self.std = MIMICIV_8CHANNEL_SD
        else:
            self.mean = MIMICIV_12CHANNEL_MEAN
            self.std = MIMICIV_12CHANNEL_SD
        
    def get_records(self, **kwargs):
        data_dir = self.data_dir
        records_files = data_dir.glob('files/*/RECORDS')
        record_list = []
        for recfile in records_files:            
            parent = recfile.parent
            with open(recfile, 'rt') as fp:
                recs = fp.readlines()
                record_list += [(parent/it.strip()) for it in recs]
        record_list.sort()
        id_list = [f'{it.parent.parent.stem}-{it.parent.stem}' 
                   for it in record_list]
        record_list = [it.as_posix() for it in record_list]
        return id_list, record_list
       
    def __len__(self):
        return len(self.id_list)

    
class EcgPtbxlData(object):
    """PTB-XL ECG data
    
    Parameters
    ----------
    data_dir : str
        path to directory containing .pbz2 WaveformSequenc data files
    ykey, xkey : list
        name of target, predictor variables in data structure
    xkey_name : list
        protocol short name, ('rest' or 'stress')
    scale_y : bool
        standardize target using training mean and SD (for continuous targets)
    ykey_scale : list
        list indicating whether to standardize (1) or not (0) the i^th target
        variable in `ykey`
    config : dict
        dict of options
    use_split : str ['train']
        select data subset, one of ['train', 'val', 'test', 'all]
    ecg_filter : bool [False]
        apply ECG (zero phase FIR) filters for removal of baseline wander
        and high frequency noise in data transform
    notch_freq : float [50]
        notch filter frequency in Hz for powerline filtering (European default)
    bandpass_freq : tuple [(0.01, 150)]
        pass band frequencies in Hz (defaults suggested from "GE Healthcare
        Marquette (tm) 12SL(tm) ECG Analysis Program Physician's Guide
        2056246-002C")
    recode_sex : bool [True]
        recode -> Female=0, Male=1
    """
    def __init__(self, data_dir=None, ykey=[], xkey=[None],
                 xkey_name=['rest'], ykey_scale=[],
                 config={}, use_split='train',
                 ecg_filter=False, sampling_rate=500., 
                 notch_freq=50., bandpass_freq=(0.01, 150),
                 scale_y=False, recode_sex=True,
                 **kwargs):
        self.auto_ecg_tasks = ['all_scp', 'diagnostic', 'subdiagnostic', 
                               'superdiagnostic', 'form', 'rhythm']
        tasks = self.auto_ecg_tasks + ['age', 'sex', 'bmi']
        assert (len(ykey) > 0) and (len(set(ykey) & set(tasks)) > 0), f'unknown task(s) {ykey}'
        init_wfdb(data_dir)
        if use_split == 'all':
            which = 'all'
        elif use_split == 'test':
            which = 'test'
        elif use_split == 'val':
            which = 'val'
        else:
            which = 'train'
        self.use_split = use_split            
        self.data_dir = data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError('`data_dir` not found')       
        self.ykey = ykey
        self.scale_y = scale_y
        self.ykey_scale = ykey_scale
        self.config = config
        self.recode_sex = recode_sex
        id_list, record_list, label_data, classes = self.get_records(which)
        self.id_list = id_list
        self.record_list = record_list
        self.label_data = label_data
        self.classes = classes
        if len(ykey) > 1 and 'task' in config:
            assert type(config['task']) in (list, tuple)
            assert len(config['task']) == len(ykey)
        self.xkey = xkey
        self.xkey_name = xkey_name
        if len(xkey) != len(xkey_name):
            raise ValueError(f'xkey length must equal xkey_name length')
        
        self.waveforms = {}
        self.ecg_filter = ecg_filter
        self.sampling_rate = sampling_rate  # Hz
        self.notch_freq = notch_freq
        self.bandpass_freq = bandpass_freq  # Hz
        self.trim_channels = kwargs.get('trim_channels', False)
        if self.trim_channels:
            self.mean = PTBXL_8CHANNEL_MEAN
            self.std = PTBXL_8CHANNEL_SD
        else:
            self.mean = PTBXL_12CHANNEL_MEAN
            self.std = PTBXL_12CHANNEL_SD
        self.transform = kwargs.get('transform', {})
              
    def get_records(self, which):
        data_dir = self.data_dir
        ykey = self.ykey
        config = self.config
        scale_y = self.scale_y
        dfp = read_ptbxl_database(data_dir)        
        dfs = pd.read_csv(data_dir/PTBXL_SCP_FILE, index_col=0)
        classes = []
        if len(ykey) > 0:
            if ykey[0] in self.auto_ecg_tasks:  # PTB-XL task
                task_name = ykey[0]
                dfp = compute_label_aggregations(dfp, task_name, dfs)
                dfp, y, classes = select_data_for_ptbxl_task(dfp, task_name)
            else:
                for yk in ykey:
                    if yk not in dfp:
                        raise ValueError(
                            f'{PTBXL_DATABASE_FILE} missing label {yk}')
                if self.recode_sex:
                    print('recoded sex variable (F/M -> 0/1)')
                    dfp['sex'] = (dfp['sex'].values + 1) % 2
                idx = select_data_no_nan(dfp, ykey)
                dfp, y = dfp[idx], dfp.loc[idx, ykey].values
                if ('task' in config and config['task'] == 'regression' and 
                    scale_y):               
                    # scaling constants for regression and scalar covariates
                    fn_stats = data_dir/PTBXL_LABEL_AUX_STATS
                    y_train_loc, y_train_scale = load_scale_constants(
                        ykey, file_in=fn_stats)
                    self.y_train_loc = y_train_loc
                    self.y_train_scale = y_train_scale
        else:
            y = dfp.shape[0] * [None]  # SSL or testing only
          
        if which == 'all':
            split = (dfp.strat_fold <= 10).values
        elif which == 'test':
            split = (dfp.strat_fold == 10).values
        elif which == 'val':
            split = (dfp.strat_fold == 9).values
        else:
            train_folds = np.arange(1,9)
            split = (dfp.strat_fold.isin(train_folds)).values
        id_list = dfp[split].index.values
        record_list = dfp[split].filename_hr.values
        label = y[split]
        return id_list, record_list, label, classes
       
    def __len__(self):
        return len(self.id_list)

    
class EcgPtbxlAllScpData(EcgPtbxlData):
    """PTB-XL all_scp multilabel (71) task 
    """
    def __init__(self, ykey=['all_scp'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))


class EcgPtbxlDiagnosticData(EcgPtbxlData):
    """PTB-XL diagnostic multilabel (44) task 
    """
    def __init__(self, ykey=['diagnostic'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))


class EcgPtbxlSubDiagnosticData(EcgPtbxlData):
    """PTB-XL subdiagnostic multilabel (23) task
    """
    def __init__(self, ykey=['subdiagnostic'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))


class EcgPtbxlSuperDiagnosticData(EcgPtbxlData):
    """PTB-XL superdiagnostic multilabel (5) task 
    """
    def __init__(self, ykey=['superdiagnostic'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))


class EcgPtbxlFormData(EcgPtbxlData):
    """PTB-XL form multilabel (19) task 
    """
    def __init__(self, ykey=['form'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))


class EcgPtbxlRhythmData(EcgPtbxlData):
    """PTB-XL rhythm multilabel (12) task 
    """
    def __init__(self, ykey=['rhythm'], **kwargs):
        super().__init__(ykey=ykey, **kwargs)
        self.config.update(dict(task='classification'))

        
data_factory = dict(mimic=EcgMimicIvData,
                    ptbxl_allscp=EcgPtbxlAllScpData,
                    ptbxl_diag=EcgPtbxlDiagnosticData,
                    ptbxl_subdiag=EcgPtbxlSubDiagnosticData,
                    ptbxl_superdiag=EcgPtbxlSuperDiagnosticData,
                    ptbxl_form=EcgPtbxlFormData,
                    ptbxl_rhythm=EcgPtbxlRhythmData,
                    )

