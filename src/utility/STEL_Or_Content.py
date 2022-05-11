"""
    STEL-specific calls, needs access to https://github.com/nlpsoc/STEL
"""
import time

import logging

# needs access to https://github.com/nlpsoc/STEL
from global_identifiable import include_STEL_project
include_STEL_project()
import eval_style_models
from set_for_global import ALTERNATIVE12_COL, ALTERNATIVE11_COL, ANCHOR2_COL, CORRECT_ALTERNATIVE_COL


def get_STEL_Or_Content_from_STEL(pd_stel_instances):
    """
        create the STEL-Or-Content task from oroginal STEl instances

        :param pd_stel_instances: pandas dataframe of original STEL instances
    """
    for row_id, row in pd_stel_instances.iterrows():
        if row[CORRECT_ALTERNATIVE_COL] == 1:
            # S1-S2 is correct order, i.e., style of A1 and S1 is the same
            pd_stel_instances.at[row_id, ALTERNATIVE12_COL] = row[ANCHOR2_COL]
        else:
            # S2-S1 is correct order, i.e., style of A1 and S2 is the same
            pd_stel_instances.at[row_id, ALTERNATIVE11_COL] = row[ANCHOR2_COL]
    return pd_stel_instances


def test_model_on_STEL(model, model_name, results_folder):
    """
        test on STEL
        :param model: model to test
        :param model_name: name of model for saving results
        :param results_folder: where to save results to

    """
    logging.info("testing model on STEl ... ")
    import os
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dim_path = [cur_dir + '/../../../STEL/Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']
    stel_pred_filename = f"{results_folder}/03_STEL_single-predictions_{model_name}__{time.time()}.tsv"
    result_dict = eval_style_models.eval_sim(output_folder=results_folder, style_objects=[model],
                                             stel_dim_tsv=dim_path, single_predictions_save_path=stel_pred_filename)
    pd_stel_instances = result_dict["stel_tasks"]
    # test on "content-adapted" STEL (STEL-Or-Content in the paper), i.e.,
    #   A, SA have same topic but distinct style and A, DA have distinct topic but same style
    pd_stel_instances = get_STEL_Or_Content_from_STEL(pd_stel_instances)
    ta_stel_pred_filename = f"{results_folder}/03_t-a-STEL_single-predictions_{model_name}__{time.time()}.tsv"
    eval_style_models.eval_sim(output_folder=results_folder, style_objects=[model],
                               eval_on_triple=True, stel_instances=pd_stel_instances,
                               single_predictions_save_path=ta_stel_pred_filename)
