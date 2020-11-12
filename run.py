from pandas import read_csv
from os.path import join
from VariableSelection.feature_selector import FeatureSelector
from MissingValuesHandler.missing_data_handler import RandomForestImputer
from DataTypeIdentifier.data_type_identifier import DataTypeIdentifier
from kmeans_functions import (compute_sse, compute_silhouette, plot_scores, 
                              get_optimal_n_clusters, plot_clusters, 
                              plot_clusters_profile)

"""/////////////////////////////////////////////////////////////////////////
****************************************************************************
*************************** 1- TARGET CANDIDATE ****************************
****************************************************************************
////////////////////////////////////////////////////////////////////////////
"""

"""
############################################
############  MAIN OBJECT  ################
############################################
"""
data = read_csv(join("loan_approval.csv"), sep=";") 
feature_selector = FeatureSelector(data)

"""
############################################
############ DATA RETRIEVAL  ###############
############################################
"""
oob_matrix = feature_selector.get_oob_score_matrix(n_estimators=50, 
                                                   additional_estimators=10, 
                                                   min_samples_split=30)
#target variable candidate
means = (oob_matrix.sum() - 1)/(oob_matrix.shape[0] - 1)
target_candidate_name = means.idxmax()

"""
############################################
############## OOB MATRIX   ################
############################################
"""
feature_selector.show_matrix_graph(oob_matrix, "Out of bag error matrix")


"""/////////////////////////////////////////////////////////////////////////
****************************************************************************
******** 2- MISSING DATA, DISTANCE MATRIX AND MDS COORDINATES***************
****************************************************************************
////////////////////////////////////////////////////////////////////////////
"""

"""
############################################
############  MAIN OBJECT  ################
############################################
"""
data = data.drop("Loan_Status", axis=1)
random_forest_imputer = RandomForestImputer(data=data,
                                            target_variable_name=target_candidate_name,
                                            training_resilience=3, 
                                            n_iterations_for_convergence=5,
                                            forbidden_features_list=["Credit_History"],
                                            ordinal_features_list=[])

"""
############################################
############### RUN TIME ###################
############################################
"""
#Setting the ensemble model parameters: it could be a random forest regressor or classifier
random_forest_imputer.set_ensemble_model_parameters(n_estimators=50, 
                                                    additional_estimators=10)

#Launching training and getting our new dataset
new_data = random_forest_imputer.train()

"""
############################################
########## DATA RETRIEVAL ##################
############################################
"""
final_proximity_matrix = random_forest_imputer.get_proximity_matrix()
final_distance_matrix  = random_forest_imputer.get_distance_matrix()

"""
############################################
########## MDS COORDINATES AND PLOT    #####
############################################
"""
mds_coordinates = random_forest_imputer.get_mds_coordinates(n_dimensions=2, 
                                                            distance_matrix=final_distance_matrix)
random_forest_imputer.show_mds_plot(mds_coordinates, plot_type="2d")


"""/////////////////////////////////////////////////////////////////////////
****************************************************************************
************************* 3- K-MEANS CLUSTERING ****************************
****************************************************************************
////////////////////////////////////////////////////////////////////////////
"""

"""
###################################################################
########## 1- Choosing optimal number of clusters K ###############
###################################################################
"""
number_clusters = 11
kmeans_kwargs = {"init": "k-means++", "max_iter": 300}
#Choosing the optimal number of clusters with the ELBOW method
sse_results = compute_sse(coordinates=mds_coordinates,
                  n_clusters=number_clusters,
                  kmeans_kwargs=kmeans_kwargs) 
sse = sse_results["sse"] 
models_sse = sse_results["models"]     
plot_scores(scores=sse, n_clusters=number_clusters, type_="sse", title="SSE")
optimal_n_clusters_sse = get_optimal_n_clusters(n_clusters=number_clusters, 
                                                scores=sse,
                                                type_="sse")


#Choosing the optimal number of clusters with the SILHOUETTE method
silhouette_results = compute_silhouette(mds_coordinates, 
                                       n_clusters=number_clusters, 
                                       kmeans_kwargs=kmeans_kwargs)
silhouette_scores = silhouette_results["silhouette_scores"]
models_silhouette = silhouette_results["models"]
plot_scores(scores=silhouette_scores, 
            type_="silhouette",
            title="Silhouette Coefficient", 
            n_clusters=number_clusters,
            path_to_save=join("graphs"))
optimal_n_clusters_sil = get_optimal_n_clusters(n_clusters=number_clusters,
                                                scores=silhouette_scores,
                                                type_="silhouette")

#Choosing K-means model
kmeans_sse = models_sse[optimal_n_clusters_sse]
kmeans_sil = models_silhouette[optimal_n_clusters_sil]

"""
###########################################
########## 2- Clusters plot ###############
###########################################
"""
kmeans = kmeans_sse
colors = ["blue", "green", "yellow","orange"]
predictions = kmeans.predict(mds_coordinates)
plot_clusters(model=kmeans,
              coordinates=mds_coordinates,
              predictions=predictions,
              colors=colors,
              figsize=(8,8),
              path_to_save=join("graphs"))


"""
###########################################
########## 3- Clusters profile ############
###########################################
"""
dtype_predictions = DataTypeIdentifier().predict(new_data)
plot_clusters_profile(data=new_data, 
                      model=kmeans, 
                      predictions=predictions, 
                      dtype_predictions=dtype_predictions, 
                      figsize=(50,200), 
                      title_fontsize=100, 
                      fontsize=50, 
                      path_to_save=join("graphs"))
 