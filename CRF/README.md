# CRF model

Once the .py files in _/Attribute retrieval_ have been run, the CRF model is ready to be used.

1. `pip install -r requirements.txt` 
2. **Run _data_prep.py_.** This prepares the data for 5 separate experiments. Expected running time: 10s*.

3. **For each experiment:**
* Open the relevant directory (_/CRF/experiment_i_xyz_)
* Run _grid_search.py_  for hyper-parameter tuning. This results in 3 separate files in
 _/CRF/experiment_i_xyz/results/_. The files are pickle files, containing lists of tuples `(f1_score, parameters)`.
 Expected running time: 2-4 hours each*.
 * Determine the optimal hyper-parameters by parsing the 3 files. This has already been done and the optimal parameters
 are hard-coded into _cross_validate.py_.
 * Run _cross_validate.py_. The final results are printed to stdout and are included in a comment at the end
 of the _cross_validate.py_ file. Expected running time: 0.5 - 2 hours *, depending on the experiment.
 
Class        | precision     | recall    |  F1
------------ | ------------- | --------- | ----
per          | x.xx          | y.yy      | z.zz
org          | x.xx          | y.yy      | z.zz
misc         | x.xx          | y.yy      | z.zz
loc          | x.xx          | y.yy      | z.zz
not-propn    | x.xx          | y.yy      | z.zz


\* Expected running times on an Intel i7-4600U CPU and 8GB of RAM
