import numpy as np

from aux import Generic


def fit_h(x, y, wdw_d, method, params):
    
    if method == 'built-in':
        # sklearn built-in method

        model = params['model']

        x_xtd = make_extended_predictor_matrix(
            vs={'x': x}, windows={'x': wdw_d}, order=['x'])

        valid = np.all(~np.isnan(x_xtd), axis=1) & (~np.isnan(y))

        rgr = model()
        rgr.fit(x_xtd[valid], y[valid])

        h = rgr.coef_.copy()
        icpt = rgr.intercept_
        
    elif method == 'wiener':
        
        raise NotImplementedError('Weiner fitting not implemented yet.')
        
    else:
        
        raise ValueError('Method not recognized.')
        
    return h, icpt
            

def fit_h_train_test(trial, x_name, y_name, wdw, train_len, test_len, method, params, C):
    """
    Fit a filter mapping one trial variable to another.
    
    :return: 
        FitResult object with attributes:
            trial_name
            
            x_name
            y_name
            
            x_wdw
            x_wdw_dsct
            
            train
            test
            
            t
            x
            y
            
            r2_train
            y_hat_train
            
            r2_test
            y_hat_test
            
            t_h
            h
    """
    # get variables of interest
    t = getattr(trial.dl, 't')
    x = getattr(trial.dl, x_name)
    y = getattr(trial.dl, y_name)
    
    n_t = len(t)
    
    # discretize window and training/test lengths
    wdw_d = (int(round(wdw[0] / C.DT)), int(round(wdw[1] / C.DT)))
    
    train_len_d = int(round(train_len / C.DT))
    test_len_d = int(round(test_len / C.DT))
    chunk_len_d = train_len_d + test_len_d
    
    t_h = np.arange(*wdw_d) * C.DT
    
    # extract training chunks from data
    valid_start = max(-wdw_d[0], 0)
    valid_end = min(n_t-wdw_d[1]+1, n_t)
    
    assert np.all(
        (~np.isnan(x[valid_start:valid_end]))
        & (~np.isnan(y[valid_start:valid_end])))
    
    n_valid = valid_end - valid_start
    n_chunks = int((n_valid + test_len_d) / chunk_len_d)
    
    train = np.zeros(n_t, bool)
    test = np.zeros(n_t, bool)
    
    x_chunks = np.nan * np.zeros((n_chunks, train_len_d))
    y_chunks = np.nan * np.zeros((n_chunks, train_len_d))
    
    for chunk_ctr in range(n_chunks):
        
        chunk_start = valid_start + (chunk_ctr * chunk_len_d)
        chunk_end = chunk_start + train_len_d
        
        x_chunks[chunk_ctr, :] = x[chunk_start:chunk_end]
        y_chunks[chunk_ctr, :] = y[chunk_start:chunk_end]
        
        train[chunk_start:chunk_end] = True
        test[chunk_end:chunk_end + test_len_d] = True
    
    # fit filters for each chunk
    hs = np.nan * np.zeros((n_chunks, len(t_h)))
    icpts = np.nan * np.zeros(n_chunks)
    
    for chunk_ctr, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
        
        h, icpt = fit_h(x_chunk, y_chunk, wdw_d, method, params)
        
        hs[chunk_ctr] = h.copy()
        icpts[chunk_ctr] = icpt.copy()
        
    h_mean = hs.mean(axis=0)
    icpt_mean = icpts.mean(axis=0)
    
    # calculate R2 on test data
    x_xtd = make_extended_predictor_matrix(vs={x_name: x}, windows={x_name: wdw_d}, order=[x_name])
    y_hat = x_xtd.dot(h_mean) + icpt_mean
    
    y_hat_train = np.nan * np.zeros(n_t)
    y_hat_train[train] = y_hat[train]
    
    y_hat_test = np.nan * np.zeros(n_t)
    y_hat_test[test] = y_hat[test]
    
    r2_train = calc_r2(y, y_hat_train)
    r2_test = calc_r2(y, y_hat_test)
    
    fit_result_dict = {
        'trial_name': trial.name,
        'x_name': x_name,
        'y_name': y_name,
        'wdw': wdw,
        'wdw_d': wdw_d,
        'train': train,
        'test': test,
        'train_len': train_len,
        'test_len': test_len,
        't': t,
        'x': x,
        'y': y,
        't_h': t_h,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'y_hat_train': y_hat_train,
        'y_hat_test': y_hat_test,
        'h': h_mean,
        'icpt': icpt_mean,
    }
    
    return Generic(**fit_result_dict)


def make_extended_predictor_matrix(vs, windows, order):
    """
    Make a predictor matrix that includes offsets at multiple time points.
    For example, if vs has 2 keys 'a' and 'b', windows is {'a': (-1, 1),
    'b': (-1, 2)}, and order = ['a', 'b'], then result rows will look like:
    
        [v['a'][t-1], v['a'][t], v['b'][t-1], v['b'][t], v['b'][t+1]]
        
    :param vs: dict of 1-D array of predictors
    :param windows: dict of (start, end) time point tuples, rel. to time point of 
        prediction, e.g., negative for time points before time point of
        prediction
    :param order: order to add predictors to final matrix in
    :return: extended predictor matrix, which has shape
        (n, (windows[0][1]-windows[0][0]) + (windows[1][1]-windows[1][0]) + ...)
    """
    if not np.all([w[1] - w[0] >= 0 for w in windows.values()]):
        raise ValueError('Windows must all be non-negative.')
        
    n = len(list(vs.values())[0])
    if not np.all([v.ndim == 1 and len(v) == n for v in vs.values()]):
        raise ValueError(
            'All values in "vs" must be 1-D arrays of the same length.')
        
    # make extended predictor array
    vs_extd = []
    
    # loop over predictor variables
    for key in order:
        
        start, end = windows[key]
        
        # make empty predictor matrix
        v_ = np.nan * np.zeros((n, end-start))

        # loop over offsets
        for col_ctr, offset in enumerate(range(start, end)):

            # fill in predictors offset by specified amount
            if offset < 0:
                v_[-offset:, col_ctr] = vs[key][:offset]
            elif offset == 0:
                v_[:, col_ctr] = vs[key][:]
            elif offset > 0:
                v_[:-offset, col_ctr] = vs[key][offset:]

        # add offset predictors to list
        vs_extd.append(v_)

    # return full predictor matrix
    return np.concatenate(vs_extd, axis=1)


def calc_r2(y, y_hat):
    """
    Calculate an R^2 value between a true value
    and a prediction.
    """
    valid = (~np.isnan(y)) & (~np.isnan(y_hat))
    
    if valid.sum() == 0:
        return np.nan
    else:
        u = np.sum((y[valid] - y_hat[valid]) ** 2)
        v = np.sum((y[valid] - y_hat[valid].mean()) ** 2)
        
        return 1 - (u/v)
