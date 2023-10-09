
def run_model(all_data, job, job_dir=None):
    index_names = ['SP500TR', 'SP400TR', 'SP600TR'] if job['train_subset'] == 'SP1500' \
                    else [job['train_subset']]
    if job['train_subset'] is not None:
        data = all_data.loc[all_data.mkt_index.isin(index_names)].copy()
    else:
        data = all_data
    
    data = data.sort_values('final_datetime')
    
    model_base = job['model']
    
    is_regr = is_regressor(model_base)
    
    returns = all_data[job['train_target']]
    train_target = returns if is_regr else (returns > 0.0)*1.0
    return_target = all_data[job['return_target']]
    
    feature_data = all_data[job['features']]
    
    all_results = []
    split_results = []
    
    trained_models = []
    trained_X_data = []
    trained_y_return = []
    
    if not is_regr:
        metrics = {'mcc': matthews_corrcoef, 'bacc': balanced_accuracy_score}
    else:
        metrics = {'r2': explained_variance_score, 'mse': mean_squared_error,
                   'bacc': directional_bacc}
    
    val = job['validator']
    print(val)

    for i, (train_index, test_index, val_index) in enumerate(val.split(data)):
        # ... [rest of the loop code]

        all_results.append(result)
        split_results.append(split_result)

    if not all_results:
        return "No results to process", {}

    results = pd.concat(all_results)
    
    # ... [rest of the code after the loop]

    return results, job_results
