# --------------------Section 1: Set working directory--------------------------------------------------------------------------
setwd(dirname(file.choose()))
getwd()

# Import data file glass.csv and put relevant variables in a data frame
glass <- read.csv("glass.data", stringsAsFactors = FALSE)
head(glass)


# -----------------------Section 2: Exploring and preparing the data------------------------------------------------------------------

# set column names
colnames(glass) <- c("Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")
glass$Id <- NULL  # drop ID column
str(glass)  # Examine the structure of the data
summary(glass)  # Summary statistics

# Check for missing values
apply(glass, MARGIN = 2, FUN = function(x) sum(is.na(x)))  # No missing data check

# Box plot for raw data
boxplot(glass[, 1:10], main = "Boxplot of Features (Raw Data)", 
        xlab = "Features", ylab = "Value", col = "lightblue", 
        las = 2, border = "darkblue")

# Normalize using Min-Max
glass_mm <- as.data.frame(apply(glass[1:9], MARGIN = 2, FUN = function(x)
  (x - min(x)) / diff(range(x))))

# Normalize using Z-Score
glass_zscore <- scale(glass[1:9])

#Define the Softmax function
softmax <- function(x) {
  exp_x <- exp(x - max(x))  # Subtract max(x) for numerical stability
  return(exp_x / sum(exp_x))  # Normalize to sum to 1
}

#Apply Softmax to each feature (column-wise)
glass_softmax <- as.data.frame(apply(glass[, 1:9], MARGIN = 2, FUN = softmax))


# Box plot for Min-Max normalized data
boxplot(glass_mm, main = "Boxplot of Features (Min-Max Normalized)", 
        xlab = "Features", ylab = "Normalized Value", col = "lightgreen", 
        las = 2, border = "darkgreen")

# Box plot for Z-Score normalized data
boxplot(glass_zscore, main = "Boxplot of Features (Z-Score Normalized)", 
        xlab = "Features", ylab = "Z-Score Value", col = "lightcoral", 
        las = 2, border = "darkred")

# Box Plot for Softmax Normalization
boxplot(glass_softmax, main = "Boxplot of Features (Softmax Normalized)", 
        xlab = "Features", ylab = "Softmax Normalized Value", col = "lightgreen", 
        las = 2, border = "darkgreen")

# Correlation Matrix between independent variable and target variable
cor_matrix <- cor(glass[, 1:10])  
print("Correlation Matrix:")
print(cor_matrix)
library(corrplot)
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)

# Correlation matrix for the features
cor_matrix <- cor(glass[, 1:9])  
print("Correlation Matrix:")
print(cor_matrix)
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)


# Check if the normalization worked 
summary(glass_zscore)

# Specifying Z-score as the active normalization method 
active_normalized_data <- data.frame(glass_zscore, Type = glass$Type)


# ----------------------------Section 3: Split the data into training and testing sets (80-20 split)--------------------------------------------------------
set.seed(123)  # Set seed for reproducibility
train_index <- sample(1:nrow(glass), 0.8 * nrow(glass))  # 80% for training
glass_train <- glass[train_index, ]
glass_test <- glass[-train_index, ]

# Extract labels (target variable 'Type')
glass_train_labels <- glass_train$Type
glass_test_labels <- glass_test$Type

# Remove target variable from the features
glass_train <- glass_train[, -1]  # Removing the 'Type' column
glass_test <- glass_test[, -1]  # Removing the 'Type' column

#---------------------- Section 4 : SVM MODEL---------------------------------------------------------------------------------------------------------- 
# Train the SVM model
library(e1071)  
svm_model <- svm(glass_train, as.factor(glass_train_labels), kernel = "radial")  

#Make predictions on the test data
glass_test_pred <- predict(svm_model, glass_test)

#Evaluate model performance using confusion matrix
library(caret)
confusionMatrix(glass_test_pred, as.factor(glass_test_labels))

# Check the accuracy
accuracy <- sum(glass_test_pred == glass_test_labels) / length(glass_test_labels)
cat("SVM Model Accuracy:", round(accuracy * 100, 2), "%\n")


# Load required libraries for evaluation
library(pROC)   
library(caret)   
library(dplyr)  

# Get predicted probabilities 
svm_probs <- attr(predict(svm_model, glass_test, decision.values = TRUE), "decision.values")

# Convert test labels to factor (ensure same levels as predictions)
glass_test_labels <- as.factor(glass_test_labels)


# Macro and Weighted F1 
cm <- confusionMatrix(glass_test_pred, glass_test_labels)
table(glass_test_labels)  # True class distribution
table(glass_test_pred)    # Predicted classes

# Extract macro F1 directly from caret
macro_f1 <- cm$byClass[, "F1"] %>% mean(na.rm = TRUE)


# Calculate weighted F1 
cm <- confusionMatrix(glass_test_pred, glass_test_labels)
class_f1 <- cm$byClass[, "F1"]
class_weights <- table(glass_test_labels) / length(glass_test_labels)

# Handle NA values 
class_f1[is.na(class_f1)] <- 0  # Treat missing F1 as 0
weighted_f1 <- sum(class_f1 * class_weights, na.rm = TRUE)

cat("Macro F1:", round(macro_f1, 3), "\n")
cat("Weighted F1:", round(weighted_f1, 3), "\n")

# Multiclass ROC-AUC (One-vs-Rest) 
# Binarize the true labels
y_true_bin <- model.matrix(~ 0 + glass_test_labels)
colnames(y_true_bin) <- levels(glass_test_labels)

# Calculate AUC for each class
auc_scores <- sapply(1:ncol(y_true_bin), function(i) {
  roc_obj <- roc(
    response = y_true_bin[, i],
    predictor = svm_probs[, i]
  )
  auc(roc_obj)
})

# Macro-average AUC
macro_auc <- mean(auc_scores)
cat("Macro AUC (OvR):", round(macro_auc, 3), "\n")


# hyperparameter tuning for SVM model using tune.svm()
svm_tune <- tune.svm(x = glass_train, y = as.factor(glass_train_labels), 
                     kernel = "radial", 
                     cost = 10^(-2:2), 
                     gamma = 10^(-2:2))

# Print best parameters from tuning
cat("Best cost:", svm_tune$best.parameters$cost, "\n")
cat("Best gamma:", svm_tune$best.parameters$gamma, "\n")


# Train SVM model again with the best parameters
svm_tuned_model <- svm(glass_train, as.factor(glass_train_labels), 
                       kernel = "radial", 
                       cost = svm_tune$best.parameters$cost, 
                       gamma = svm_tune$best.parameters$gamma)

# Make predictions on the test data with the tuned model
glass_test_pred_tuned <- predict(svm_tuned_model, glass_test)

# Evaluate model performance using confusion matrix
cm_tuned <- confusionMatrix(glass_test_pred_tuned, as.factor(glass_test_labels))

# Print confusion matrix
print(cm_tuned)

# Check the accuracy of the tuned model
accuracy_tuned <- sum(glass_test_pred_tuned == glass_test_labels) / length(glass_test_labels)
cat("Tuned SVM Model Accuracy:", round(accuracy_tuned * 100, 2), "%\n")

# Evaluate model performance using confusion matrix
cm_tuned <- confusionMatrix(glass_test_pred_tuned, as.factor(glass_test_labels))

# Calculate Macro and Weighted F1 scores for the tuned model
macro_f1_tuned <- cm_tuned$byClass[, "F1"] %>% mean(na.rm = TRUE)
weighted_f1_tuned <- cm_tuned$byClass[, "F1"] %*% (table(glass_test_labels) / length(glass_test_labels))

# Calculate weighted F1 
class_f1_tuned <- cm_tuned$byClass[, "F1"]
class_weights_tuned <- table(glass_test_labels) / length(glass_test_labels)

# Handle NA values 
class_f1_tuned[is.na(class_f1_tuned)] <- 0  # Treat missing F1 as 0
weighted_f1_tuned <- sum(class_f1_tuned * class_weights_tuned, na.rm = TRUE)

cat("Weighted F1, tuned):", round(weighted_f1_tuned, 3), "\n")
cat("Macro F1 (tuned):", round(macro_f1_tuned, 3), "\n")

# Multiclass ROC-AUC (One-vs-Rest) for the tuned model
# Binarize the true labels for the tuned model
y_true_bin_tuned <- model.matrix(~ 0 + glass_test_labels)
colnames(y_true_bin_tuned) <- levels(glass_test_labels)

# Get predicted probabilities for the tuned model 
svm_probs_tuned <- attr(predict(svm_tuned_model, glass_test, decision.values = TRUE), "decision.values")

# Calculate AUC for each class for the tuned model
auc_scores_tuned <- sapply(1:ncol(y_true_bin_tuned), function(i) {
  roc_obj <- roc(
    response = y_true_bin_tuned[, i],
    predictor = svm_probs_tuned[, i]
  )
  auc(roc_obj)
})

# Macro-average AUC for the tuned model
macro_auc_tuned <- mean(auc_scores_tuned)
cat("Macro AUC (OvR, tuned):", round(macro_auc_tuned, 3), "\n")




# -----------------Section 5 : Random Forest Model------------------------------------------------------------------------------------------------- 
# Load Required Libraries
install.packages("randomForest")
library(randomForest)

# Convert response to factor for classification
glass$Type <- as.factor(glass$Type)

# Split into training and test sets
set.seed(123)
train_index <- createDataPartition(glass$Type, p = 0.8, list = FALSE)
glass_train <- glass[train_index, ]
glass_test <- glass[-train_index, ]

# Train Random Forest Model 
set.seed(123)
rf_model <- randomForest(Type ~ ., data = glass_train, ntree = 500, mtry = 3, importance = TRUE)

# Print model summary
print(rf_model)
varImpPlot(rf_model)

# Predict on test set
rf_pred <- predict(rf_model, newdata = glass_test)

# Ensure both are factors with the same levels
rf_pred <- factor(rf_pred, levels = levels(glass_test$Type))
glass_test$Type <- factor(glass_test$Type, levels = levels(glass_test$Type))

# Compute confusion matrix
conf_mat <- confusionMatrix(rf_pred, glass_test$Type)
print(conf_mat)

# Get predicted probabilities
rf_probs <- predict(rf_model, glass_test, type = "prob")

# Convert test labels to factor (ensure same levels as predictions)
glass_test_labels <- glass_test$Type

# Macro and Weighted F1 
cm_rf <- confusionMatrix(rf_pred, glass_test_labels)

# Extract macro and weighted F1 directly from caret
macro_f1_rf <- mean(cm_rf$byClass[, "F1"], na.rm = TRUE)
weighted_f1_rf <- cm_rf$byClass[, "F1"] %*% (table(glass_test_labels) / length(glass_test_labels))

cat("\nRandom Forest Performance:\n")
cat("Macro F1:", round(macro_f1_rf, 3), "\n")
cat("Weighted F1:", round(weighted_f1_rf, 3), "\n")

# Multiclass ROC-AUC (One-vs-Rest) 
# Binarize the true labels
y_true_bin <- model.matrix(~ 0 + glass_test_labels)
colnames(y_true_bin) <- levels(glass_test_labels)

# Calculate AUC for each class
auc_scores_rf <- sapply(1:ncol(y_true_bin), function(i) {
  roc_obj <- roc(
    response = y_true_bin[, i],
    predictor = rf_probs[, i]
  )
  auc(roc_obj)
})

# Macro-average AUC
macro_auc_rf <- mean(auc_scores_rf)
cat("Macro AUC (OvR):", round(macro_auc_rf, 3), "\n")

# Hyperparameter Tuning for Random Forest Model 
# Set up cross-validation
ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

# Define tuning grid for mtry
tune_grid <- expand.grid(mtry = 1:7)  

# Train Random Forest with tuning
set.seed(123)
rf_tuned <- train(
  Type ~ .,
  data = glass_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = tune_grid,
  importance = TRUE,
  ntree = 1000
)

# View results of tuned model
print(rf_tuned)
plot(rf_tuned)

# Make predictions on the test data with the tuned model
rf_tuned <- predict(rf_model, glass_test, type = "prob")

# Convert test labels to factor (ensure same levels as predictions)
glass_test_labels <- glass_test$Type

# Convert probabilities to predicted class labels
rf_tuned_labels <- colnames(rf_probs)[apply(rf_probs, 1, which.max)]
rf_tuned_labels <- factor(rf_tuned_labels, levels = levels(glass_test_labels))

# calculate confusion matrix with predicted labels and true labels
cm_rf_tuned <- confusionMatrix(rf_tuned_labels, glass_test_labels)


# Extract macro and weighted F1 scores for tuned model
macro_f1_rf <- mean(cm_rf_tuned$byClass[, "F1"], na.rm = TRUE)
weighted_f1_rf <- sum(cm_rf_tuned$byClass[, "F1"] * (table(glass_test_labels) / length(glass_test_labels)))

# Output of results for tuned model 
cat("\nRandom Forest Performance:\n")
cat("Macro F1:", round(macro_f1_rf, 3), "\n")
cat("Weighted F1:", round(weighted_f1_rf, 3), "\n")


# Multiclass ROC-AUC (One-vs-Rest) for tuned model
# Binarize the true labels
y_true_bin <- model.matrix(~ 0 + glass_test_labels)
colnames(y_true_bin) <- levels(glass_test_labels)

# Calculate AUC for each class
auc_scores_rf <- sapply(1:ncol(y_true_bin), function(i) {
  roc_obj <- roc(
    response = y_true_bin[, i],
    predictor = rf_probs[, i]
  )
  auc(roc_obj)
})

# Macro-average AUC fo tuned model 
macro_auc_rf <- mean(auc_scores_rf)
cat("Macro AUC (OvR):", round(macro_auc_rf, 3), "\n")

#-----------------Cleanup-------------------------------------------------------------------------------------------------------------------------
rm(list=ls())
dev.off()
