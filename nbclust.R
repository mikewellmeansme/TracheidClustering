library(NbClust)

data_A = read.csv(file='input/quotient_deviation_df_A_CLASSIFIED.csv', sep=";", dec=",")

data_B = read.csv(file='input/quotient_deviation_df_B_CLASSIFIED.csv', sep=";", dec=",")

res_A <- NbClust(data_A, min.nc = 3, max.nc = 20, method = "kmean")

res_B <- NbClust(data_B, min.nc = 3, max.nc = 20, method = "kmean")
