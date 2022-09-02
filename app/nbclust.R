library(NbClust)

data_A = read.csv(file='input/quotient_deviation_df_A_CLASSIFIED.csv', sep=";", dec=",")

data_B = read.csv(file='input/quotient_deviation_df_B_CLASSIFIED.csv', sep=";", dec=",")

res_A <- NbClust(data_A, min.nc = 3, max.nc = 20, method = "kmean")

res_B <- NbClust(data_B, min.nc = 3, max.nc = 20, method = "kmean")

data_A_Diam = read.csv(file='input/d_cwt_separately/A_Diam.csv')
data_A_CWT = read.csv(file='input/d_cwt_separately/A_CWT.csv')
data_B_Diam = read.csv(file='input/d_cwt_separately/B_Diam.csv')
data_B_CWT = read.csv(file='input/d_cwt_separately/B_CWT.csv')


res_A_Diam <- NbClust(data_A_Diam, min.nc = 2, max.nc = 20, method = "kmean")
res_A_CWT <- NbClust(data_A_CWT, min.nc = 2, max.nc = 20, method = "kmean")
res_B_Diam <- NbClust(data_B_Diam, min.nc = 2, max.nc = 20, method = "kmean")
res_B_CWT <- NbClust(data_B_CWT, min.nc = 2, max.nc = 20, method = "kmean")


data_kaz_A = read.csv(file='output/KAZ_obects_for_clustering_A.csv', sep=",", dec=".")

NbClust(data_kaz_A, min.nc = 2, max.nc = 20, method = "kmean")
