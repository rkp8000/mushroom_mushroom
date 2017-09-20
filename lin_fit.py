import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from aux import make_extended_predictor_matrix


class ClassifierResult(object):
    
    def __init__(self, trial_id, preds, windows, valid, states, states_pred, coefs, clf):
        self.trial_id = trial_id
        self.preds = preds
        self.windows = windows
        self.valid = valid
        self.states = states
        self.states_pred = states_pred
        self.coefs = coefs
        self.clf = clf
        
        # get number of valid time points
        self.n_valid = valid.sum()
        
        # get fraction of valid time points in walking state
        self.walk_frac = np.sum(states[valid] == 'W') / self.n_valid
        
        # get accuracy (fraction of correctly labeled valid states)
        if states_pred is not None:
            self.acc = np.sum(states_pred[valid] == states[valid]) / self.n_valid
        else:
            self.acc = np.nan
            
            
def classify_states(trial, preds, windows):
    """
    Fit a classifier predicting a trial's paused vs. walking states
    using activity from a set of neural compartments.
    
    :param trial: trial object
    :param windows: dict of windows to use for each predictive dan
        (keys are DAN names, vals are tuples of (start, end) time
            points relative to time point of prediction)
            
    :return: ClassifierResult
    """
    
    try:
        trial.dl
    except:
        trial.dl = DataLoader(trial, vel_filt=None)
    
    states = trial.dl.state
    vs = {pred: getattr(trial.dl, pred) for pred in preds}
    
    # make extended dan predictor matrix
    vs_extd = make_extended_predictor_matrix(vs, windows, order=preds)
    
    # make valid mask (not nan predictors and not ambiguous state)
    valid = np.all(~np.isnan(vs_extd), axis=1) & (states != 'A')
    
    if len(set(states[valid])) == 2:
        # at least two states
        
        # fit classifier
        clf = LogisticRegression(n_jobs=-1)
        clf.fit(vs_extd[valid], states[valid])
        
        # make state predictions
        states_pred = np.repeat('', len(states))
        states_pred[valid] = clf.predict(vs_extd[valid])
        
        # get coefficients
        window_lens = [windows[pred][1] - windows[pred][0] for pred in preds]
        splits = np.split(clf.coef_[0], np.cumsum(window_lens)[:-1])
        
        coefs = {pred: s for pred, s in zip(preds, splits)}
        
    else:
        # fewer than two states
        clf = None
        states_pred = None
        coefs = None
        
    result = ClassifierResult(
        trial_id=trial.id, preds=preds, windows=windows, valid=valid,
        states_pred=states_pred, states=states, coefs=coefs, clf=clf)
    
    return result


class RegressionResult(object):
    
    def __init__(self, trial_id, targ, preds, windows, valid, ys, ys_pred, coefs, rgr, details=None):
        self.trial_id = trial_id
        self.targ = targ
        self.preds = preds
        self.windows = windows
        self.valid = valid
        self.ys = ys
        self.ys_pred = ys_pred
        self.coefs = coefs
        self.rgr = rgr

        if details is None:
            details = {}
            
        self.details = details
        
        self.n_valid = valid.sum()

        if valid.sum() > 0:
            # calc R^2 val from truth and prediction
            self.err = ys_pred - ys
            self.r2 = (np.nanvar(ys) - np.nanvar(self.err)) / np.nanvar(ys)
        else:
            self.err = np.repeat(np.nan, len(ys))
            self.r2 = np.nan


def regress(trial, targ, preds, windows, valid='all'):
    """
    Predict one trial variable using other trial variables through a linear regression
    model.
    
    :param trial: trial object
    :param targ: target to predict
    :param preds: order of predictors to use
    :param windows: dict of windows to use for each predictor
        (keys are predictor names, vals are tuples of (start, end) time
            points relative to time point of prediction)
    :param valid: mask over points to include in prediction
            
    :return: RegressionResult
    """
    
    if not all([pred in windows for pred in preds]):
        raise KeyError('One window must be provided for each predictor.')
        
    xs = {pred: getattr(trial.dl, pred) for pred in preds}
    ys = getattr(trial.dl, targ)
    
    # make extended dan predictor matrix
    xs_extd = make_extended_predictor_matrix(xs, windows, order=preds)
    
    if valid == 'all':
        valid = np.repeat(True, len(ys))
    elif valid == 'none':
        valid = np.repeat(False, len(ys))
        
    # edit valid to ignore any rows with nans
    if valid.sum() > 0:
        valid_ = np.all(~np.isnan(xs_extd), 1) & (~np.isnan(ys))
        valid = valid & valid_
        
    ys = ys.copy()
    ys[~valid] = np.nan
    
    if valid.sum() > 0:
        # at least one valid time point
        # fit regression
        rgr = LinearRegression(n_jobs=-1)
        rgr.fit(xs_extd[valid], ys[valid])
        
        # make target predictions
        ys_pred = np.repeat(np.nan, len(ys))
        ys_pred[valid] = rgr.predict(xs_extd[valid])
        
        # get coefficients
        window_lens = [windows[pred][1] - windows[pred][0] for pred in preds]
        splits = np.split(rgr.coef_, np.cumsum(window_lens)[:-1])
        
        coefs = {pred: s for pred, s in zip(preds, splits)}
        
    else:
        # no valid time points
        rgr = None
        ys_pred = None
        coefs = None
        
    result = RegressionResult(
        trial_id=trial.id, targ=targ, preds=preds, windows=windows, valid=valid,
        ys=ys, ys_pred=ys_pred, coefs=coefs, rgr=rgr)
    
    return result


class ReconstructionResult(object):
    
    def __init__(self, trial_id, windows, clf_rslt, rgr_rslts, ys, ys_pred):
        self.trial_id = trial_id
        self.windows = windows
        self.clf_rslt = clf_rslt
        self.rgr_rslts = rgr_rslts
        self.ys = ys
        self.ys_pred = ys_pred
        
        self.errs = {}
        self.r2s = {}
        
        for key in self.ys:
            if key == 'state':
                continue
            
            self.errs[key] = ys_pred[key] - ys[key]
            self.r2s[key] = (np.nanvar(ys[key]) - np.nanvar(self.errs[key])) / np.nanvar(ys[key])
        

def reconstruct(trial, windows):
    """Reconstruct a trial's state, speed, angular velocity, and air tube motion.
    
    :param trial: trial object
    :param windows: dict of windows to use for reconstruction; it should have keys
        {'state', 'speed', 'v_ang', 'ddt_air_tube'}, and corresponding values should be
        dict indicating predictor names and windows (tuple specifying start and end time point)
    :return ReconstructionResult instance
    """
    if not set(windows) == {'state', 'speed', 'v_ang', 'ddt_air_tube'}:
        raise KeyError(
            'Predictor windows must be specified for keys "state", '
            '"speed", "v_ang", and "ddt_air_tube".')
    
    preds = {k: w.keys() for k, w in windows.items()}
    
    # classify states
    clf_rslt = classify_states(trial, preds['state'], windows['state'])
    
    # create valid mask of walking states
    valid = trial.dl.states == 'W'
    
    # fit regression models for speed, v_ang, and ddt_air_tube
    rgr_rslts = {
        'speed': regress(trial, targ='speed', preds=preds['speed'], windows=windows['speed'], valid=valid),
        'v_ang': regress(trial, targ='v_ang', preds=preds['v_ang'], windows=windows['v_ang'], valid=valid),
    }
    
    if trial.expt == 'closed_loop':
        rgr_rslts['ddt_air_tube'] = regress(
            trial, targ='ddt_air_tube', preds=preds['ddt_air_tube'], windows=windows['ddt_air_tube'], valid=valid)
    elif trial.expt == 'driven_random':
        rgr_rslts['ddt_air_tube'] = regress(
            trial, targ='ddt_air_tube', preds=preds['ddt_air_tube'], windows=windows['ddt_air_tube'], valid='all')
    else:
        rgr_rslts['ddt_air_tube'] = None
        
    # make predictions
    xs = {}
    xs_extd = {}
    valid = {}
    
    for key in ['state', 'speed', 'v_ang']:
        xs[key] = {pr: getattr(trial.dl, pr) for pr in preds[key]}
        xs_extd[key] = make_extended_predictor_matrix(xs[key], windows[key], order=preds[key])
    
    if trial.expt != 'motionless':
        xs['ddt_air_tube'] = {pr: getattr(trial.dl, pr) for pr in preds['ddt_air_tube']}
        xs_extd['ddt_air_tube'] = make_extended_predictor_matrix(
            xs['ddt_air_tube'], windows['ddt_air_tube'], order=preds['ddt_air_tube'])
        
    # states
    valid['state'] = np.all(~np.isnan(xs_extd['state']), axis=1)
    states_pred = np.repeat('', len(valid['state']))
    if clf_rslt.clf is not None:
        states_pred[valid['state']] = clf_rslt.clf.predict(xs_extd['state'][valid['state']])
    else:
        states_pred[valid['state']] = 'W'
    
    # speed
    valid['speed'] = np.all(~np.isnan(xs_extd['speed']), axis=1) & (states_pred == 'W')
    speed_pred = np.repeat(np.nan, len(valid['speed']))
    # set paused speeds to 0
    speed_pred[states_pred == 'P'] = 0
    # predict speed during valid epochs
    speed_pred[valid['speed']] = rgr_rslts['speed'].rgr.predict(xs_extd['speed'][valid['speed']])
    
    # v_ang
    valid['v_ang'] = np.all(~np.isnan(xs_extd['v_ang']), axis=1) & (states_pred == 'W')
    v_ang_pred = np.repeat(np.nan, len(valid['v_ang']))
    # set paused v_ang to 0
    v_ang_pred[states_pred == 'P'] = 0
    # predict v_ang during valid epochs
    v_ang_pred[valid['v_ang']] = rgr_rslts['v_ang'].rgr.predict(xs_extd['v_ang'][valid['v_ang']])
    
    # air tube
    ddt_air_tube_pred = np.repeat(np.nan, len(trial.dl.states))
    
    if trial.expt == 'motionless':
        # everything nan since no air tube motion
        valid['ddt_air_tube'] = np.repeat(False, len(trial.dl.states))
    elif trial.expt == 'closed_loop':
        # predictors must exist and fly must be walking
        valid['ddt_air_tube'] = np.all(~np.isnan(xs_extd['ddt_air_tube']), axis=1) & (states_pred == 'W')
        # set paused ddt_air_tube to 0
        ddt_air_tube_pred[states_pred == 'P'] = 0
    elif trial.expt == 'driven_random':
        # predictors must exist but fly can be in any state
        valid['ddt_air_tube'] = np.all(~np.isnan(xs_extd['ddt_air_tube']), axis=1)
        
    if trial.expt != 'motionless':
        # predict ddt_air_tube during valid epochs
        ddt_air_tube_pred[valid['ddt_air_tube']] = rgr_rslts['ddt_air_tube'].rgr.predict(
            xs_extd['ddt_air_tube'][valid['ddt_air_tube']])
        
    rcn_rslt = ReconstructionResult(
        trial_id=trial.id, windows=windows, clf_rslt=clf_rslt, rgr_rslts=rgr_rslts,
        ys={
            'state': trial.dl.states,
            'speed': trial.dl.speed,
            'v_ang': trial.dl.v_ang,
            'ddt_air_tube': trial.dl.ddt_air_tube,
        },
        ys_pred={
            'state': states_pred,
            'speed': speed_pred,
            'v_ang': v_ang_pred,
            'ddt_air_tube': ddt_air_tube_pred,
        })
    
    return rcn_rslt