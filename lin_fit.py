import numpy as np

from aux import Generic


def fit_h(xs, y, wdws_d, order, method, params, return_model=False):
    
    if method == 'built-in':
        # sklearn built-in method

        model = params['model']

        wdw_lens = [wdws_d[x_name][1] - wdws_d[x_name][0] for x_name in order]
        
        x_xtd = make_extended_predictor_matrix(
            vs=xs, windows=wdws_d, order=order)

        valid = np.all(~np.isnan(x_xtd), axis=1) & (~np.isnan(y))
        
        if np.sum(valid) <= x_xtd.shape[1] + 1:
            
            print('Not enough valid data points.')
            
            hs = {
                x_name: np.nan * np.zeros(wdw_len)
                for x_name, wdw_len in zip(order, wdw_lens)}
            icpt = np.nan
            
            y_hat = np.nan * np.zeros(len(y))
            r2 = np.nan
            
        else:

            rgr = model()
            rgr.fit(x_xtd[valid], y[valid])

            # get concatenated filters
            h_xtd = rgr.coef_.copy()

            # get individual filters
            hs_split = np.split(h_xtd, np.cumsum(wdw_lens)[:-1])

            hs = {x_name: h for x_name, h in zip(order, hs_split)}
            icpt = rgr.intercept_
            
            y_hat = np.nan * np.zeros(len(y))
            y_hat[valid] = rgr.predict(x_xtd[valid])
            
            r2 = rgr.score(x_xtd[valid], y[valid])

    elif method == 'wiener':
        
        raise NotImplementedError('Weiner fitting not implemented yet.')
        
    else:
        
        raise ValueError('Method not recognized.')
    
    if return_model:
        return hs, icpt, y_hat, r2, rgr
    else:
        return hs, icpt, y_hat, r2


def fit_h_train_test(
        trial, x_names, y_name, wdws, train_len, test_len,
        method, params, C, allow_nans=False):
    """
    Fit a filter mapping one trial variable to another.
    
    :return: 
        FitResult object with attributes:
            trial_name
            
            x_names
            y_name
            
            wdws
            wdws_dsct
            
            train
            test
            
            t
            xs
            y
            
            r2_train
            y_hat_train
            
            r2_test
            y_hat_test
            
            t_h
            h
    """
    # get variables of interest
    ## 1D array
    t = getattr(trial.dl, 't')
    ## dict of 1D arrays with variable names as keys
    xs = {x_name: getattr(trial.dl, x_name) for x_name in x_names}
    ## 1D array
    y = getattr(trial.dl, y_name)
    
    ## number of time steps
    n_t = len(t)
    
    # discretize moving window size
    ## dict of 2-element integer tuples with variable name as keys
    wdws_d = {
        x_name: (int(round(wdw[0] / C.DT)), int(round(wdw[1] / C.DT)))
        for x_name, wdw in wdws.items()
    }
    
    # make filter time vectors
    ## dict of 1D arrays with variable names as keys
    t_hs = {
        x_name: np.arange(*wdw_d) * C.DT
        for x_name, wdw_d in wdws_d.items()
    }

    if test_len:
        # discretize training and test lengths
        train_len_d = int(round(train_len / C.DT))
        test_len_d = int(round(test_len / C.DT))
        chunk_len_d = train_len_d + test_len_d

        
        # extract training chunks from data
        
        ## get start and end points such that filter windows have no nans
        valid_start = max(np.max([-wdw_d[0] for wdw_d in wdws_d.values()]), 0)
        valid_end = min(np.min([n_t-wdw_d[1]+1 for wdw_d in wdws_d.values()]), n_t)

        if not allow_nans:
            # make sure there are no NaNs
            for x in xs.values():
                assert np.all(~np.isnan(x[valid_start:valid_end]))
            assert np.all(~np.isnan(y[valid_start:valid_end]))

        n_valid = valid_end - valid_start
        n_chunks = int((n_valid + test_len_d) / chunk_len_d)

        train = np.zeros(n_t, bool)
        test = np.zeros(n_t, bool)

        # TO BECOME: list of dicts of 1D arrays with variable names for keys
        x_chunks = []
        # TO BECOME: 2D array
        y_chunks = np.nan * np.zeros((n_chunks, train_len_d))
    
        # fill in "chunks"
        for chunk_ctr in range(n_chunks):

            chunk_start = valid_start + (chunk_ctr * chunk_len_d)
            chunk_end = chunk_start + train_len_d

            x_chunks.append(
                {x_name: x[chunk_start:chunk_end] for x_name, x in xs.items()})

            y_chunks[chunk_ctr, :] = y[chunk_start:chunk_end]

            train[chunk_start:chunk_end] = True
            test[chunk_end:chunk_end + test_len_d] = True

        # fit filters for each chunk
        hs = {x_name: [] for x_name in x_names}
        icpts = np.nan * np.zeros(n_chunks)

        for chunk_ctr, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):

            hs_chunk, icpt = fit_h(
                x_chunk, y_chunk, wdws_d, x_names, method, params)

            for x_name in x_names:
                hs[x_name].append(hs_chunk[x_name])
            icpts[chunk_ctr] = icpt

        hs_mean = {x_name: np.nanmean(hs_, 0) for x_name, hs_ in hs.items()}
        icpt_mean = np.nanmean(icpts, 0)

        # calculate R2 on test data
        x_xtd = make_extended_predictor_matrix(
            vs=xs, windows=wdws_d, order=x_names)
        h_mean_xtd = np.concatenate([hs_mean[x_name] for x_name in x_names])

        y_hat = x_xtd.dot(h_mean_xtd) + icpt_mean

        y_hat_train = np.nan * np.zeros(n_t)
        y_hat_train[train] = y_hat[train]

        y_hat_test = np.nan * np.zeros(n_t)
        y_hat_test[test] = y_hat[test]

        r2_train = calc_r2(y, y_hat_train)
        r2_test = calc_r2(y, y_hat_test)
    else:
        print('Not splitting into test/training data.')
        hs_mean, icpt_mean = fit_h(xs, y, wdws_d, x_names, method, params)
        
        # make predictions on full dataset
        x_xtd = make_extended_predictor_matrix(
            vs=xs, windows=wdws_d, order=x_names)
        h_mean_xtd = np.concatenate([hs_mean[x_name] for x_name in x_names])
        
        y_hat_train = x_xtd.dot(h_mean_xtd) + icpt_mean
        y_hat_test = np.nan * np.zeros(y_hat_train.shape)
        
        r2_train = calc_r2(y, y_hat_train)
        r2_test = np.nan
        
        train = np.ones(n_t, bool)
        test = np.zeros(n_t, bool)

    fit_result_dict = {
        'trial_name': trial.name,
        'x_names': x_names,
        'y_name': y_name,
        'wdws': wdws,
        'wdws_d': wdws_d,
        'train': train,
        'test': test,
        'train_len': train_len,
        'test_len': test_len,
        't': t,
        'xs': xs,
        'y': y,
        't_hs': t_hs,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'y_hat_train': y_hat_train,
        'y_hat_test': y_hat_test,
        'hs': hs_mean,
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
