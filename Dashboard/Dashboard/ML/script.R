library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)

#initiate an empty model

first_model <- keras_model_sequential()
#add first layer, the expected input is of shape 128 by 128 on three channels (we will be dealing with RGB images)

layer_conv_2d(first_model,filters = 64,kernel_size = 3, activation = "relu",input_shape = c(224,224,3))
layer_conv_2d(first_model,filters = 64,kernel_size = 3, activation = "relu",input_shape = c(224,224,3))
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 

layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu") 
layer_conv_2d(first_model, filters = 128, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 

layer_conv_2d(first_model, filters = 256, kernel_size = c(3, 3), activation = "relu") 
layer_conv_2d(first_model, filters = 256, kernel_size = c(3, 3), activation = "relu") 
layer_conv_2d(first_model, filters = 256, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 

layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 

layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_conv_2d(first_model, filters = 512, kernel_size = c(3, 3), activation = "relu")
layer_max_pooling_2d(first_model, pool_size = c(2, 2)) 

layer_flatten(first_model) 
layer_dense(first_model, units = 256, activation = "relu")
layer_dense(first_model, units = 128, activation = "relu")
layer_dense(first_model, units = 1, activation = "sigmoid")

summary(first_model)



#DATA PREPARATION

plot_layer_activations <- function(img_path, model, activations_layers,channels, out_path=NULL){
  
  if(is.null(out_path)){
    out_path <- getwd()
  }
  model_input_size <- c(model$input_shape[[2]], model$input_shape[[3]]) 
  
  img <- image_load(img_path, target_size =  model_input_size) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, model_input_size[1], model_input_size[2], 3)) %>%
    imagenet_preprocess_input()
  
  layer_outputs <- lapply(model$layers[activations_layers], function(layer) layer$output)
  activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
  activations <- predict(activation_model,img)
  if(!is.list(activations)){
    activations <- list(activations)
  }
  
  
  plot_channel <- function(channel,layer_name,channel_name) {
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1,
          col = terrain.colors(12),main=paste("conv. channel:",channel_name))
  }
  
  for (i in 1:length(activations)) {
    layer_activation <- activations[[i]]
    layer_name <- model$layers[[activations_layers[i]]]$name
    n_features <- dim(layer_activation)[[4]]
    for (c in channels){
      
      channel_image <- layer_activation[1,,,c]
      plot_channel(channel_image,layer_name,c)
      
    }
  } 
  
}

par(mfrow=c(1,3),mar=c(1,1,0,1),mai=c(0.1,0.1,0.3,0.1),cex=0.8)


##########################

# Data preperation #

# get all file paths of the images containing our target

subset_list <- list.files("Dashboard/Data/True/", full.names = T)

# create a data.frame with two coloumns: file paths, and the labels (1)
data_true <- data.frame(img=subset_list,lbl=rep(1L,length(subset_list)))

# get all file paths of the images containing non-targets
subset_list <- list.files("Dashboard/ML/Data/False", full.names = T)

#creating a data.frame with two coloumns: file paths, and the labels (0)
data_false <- data.frame(img=subset_list,lbl=rep(0L,length(subset_list)))

#merge both data.frames
data <- rbind(data_true,data_false)


# Prepare data for modelling

# randomly split data set into training (~75%) and validation data (~25%)
set.seed(2020)
data <- initial_split(data,prop = 0.75, strata = "lbl")

data

head(training(data)) # returns the the first few entries in the training data.frame

head(testing(data)) # returns the first few entries of the data set aside for validation

# compare the number of files in the training data, that show non-targets vs, those that
# show targets -> should be similiar
c(nrow(training(data)[training(data)$lbl==0,]), nrow(training(data)[training(data)$lbl==1,]))


#------------ COnvert data to format for CNN

## Looking at the data

#The data will be trained in batches, the first batch
#The batch will be  number of samples, rows, columns and bands

training_dataset <- tensor_slices_dataset(training(data)) #This pipeline has been processed using tfdataset package
##For inspection of the dataset one can Use a tfdataset object
## A List of all tensors
dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)
head(dataset_list)

#Two images each having the specified characteristics

subset_size <- first_model$input_shape[2:3]
subset_size

##Convert the images to a float
training_dataset <-
  dataset_map(training_dataset, function(.x)
    tf$image$decode_jpeg(tf$io$read_file(.x$img)))
#list_modify(.x, img = tf$image$decode_jpeg(tf$io$read_file(.x$img))))




##################################################
# Training the model
#################################################

compile(
  first_model,
  optimizer = optimizer_rmsprop(learning_rate = 5e-5),
  loss="binary_crossentropy",
  metrics=c("accuracy"))


###############################-----------------------------------################# 

#prepare training dataset
training_dataset <- tensor_slices_dataset(training(data))


#if you want to get a list of all tensors, you can use the as_iterator() and iterate() functions
dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)

#get input shape expected by first_model
subset_size <- first_model$input_shape[2:3]

# apply function on each dataset element: function is list_modify()
#->modify list item "img" three times:

# 1 read decode jpeg
training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$decode_jpeg(tf$io$read_file(.x$img))))

# 2 convert data type
training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)))

# 3 resize to the size expected by model
training_dataset <- 
  dataset_map(training_dataset, function(.x)
    list_modify(.x, img = tf$image$resize(.x$img, size = shape(subset_size[1], subset_size[2]))))

training_dataset <- dataset_shuffle(training_dataset, buffer_size = 10L*128)
training_dataset <- dataset_batch(training_dataset, 10L)
training_dataset <- dataset_map(training_dataset, unname)
dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)
dataset_list[[1]][[1]]
dataset_list[[1]][[1]]$shape
dataset_list[[1]][[2]]


#validation
validation_dataset <- tensor_slices_dataset(testing(data))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$decode_jpeg(tf$io$read_file(.x$img))))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32)))

validation_dataset <- 
  dataset_map(validation_dataset, function(.x)
    list_modify(.x, img = tf$image$resize(.x$img, size = shape(subset_size[1], subset_size[2]))))

validation_dataset <- dataset_batch(validation_dataset, 10L)
validation_dataset <- dataset_map(validation_dataset, unname)

## Training your model {#train}


compile(
  first_model,
  optimizer = optimizer_rmsprop(learning_rate = 5e-5),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)



diagnostics <- fit(first_model,
                   training_dataset,
                   epochs = 15,
                   validation_data = validation_dataset)

plot(diagnostics)


## Predicting with your model

predictions <- predict(first_model,validation_dataset)
head(predictions)
tail(predictions)

par(mfrow=c(1,3),mai=c(0.1,0.1,0.3,0.1),cex=0.8)
for(i in 1:3){
  sample <- floor(runif(n = 1,min = 1,max = 56))
  img_path <- as.character(testing(data)[[sample,1]])
  img <- stack(img_path)
  plotRGB(img,margins=T,main = paste("prediction:",round(predictions[sample],digits=3)," | ","label:",as.character(testing(data)[[sample,2]])))
}

save_model_hdf5(first_model,filepath = "Dashboard/ML/models/basic_model_24Sept2023_4_10PM.h5")


