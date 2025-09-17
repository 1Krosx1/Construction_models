% =========================================================================
% main_script.m (REVISED)
%
% Main script to load, process, train, and evaluate the ANN model.
% This version saves all plots as PNG files, including the training graph.
% =========================================================================

% =========================================================================
% Section 1: Setup
% =========================================================================
% Clears all variables from the workspace to start fresh.
clear; 
% Clears the command window to remove previous outputs.
clc; 
% Closes all open figures (plots).
close all;

% Sets a stable renderer to prevent blank plot issues.
set(groot, 'defaultfigurerenderer', 'painters');

% Prints a message to the console indicating the workspace is ready.
fprintf("Workspace cleared and ready.\n");


% =========================================================================
% Section 2: Data Loading, Cleaning, and Merging
% =========================================================================
% Prints a header for this section to the console.
fprintf('\n--- Section 2: Loading, Cleaning & Merging Data ---\n');

% Creates import options for the quantity data CSV file, preserving original column names.
opts_qty = detectImportOptions('Thesis Data - Architectural Quantity Cost.csv', 'VariableNamingRule', 'preserve');
% Creates import options for the unit cost data CSV file, preserving original column names.
opts_cost = detectImportOptions('Thesis Data - Achitectural Unit Cost.csv', 'VariableNamingRule', 'preserve');

% Starts a try-catch block to handle potential errors during file reading.
try
    % Reads the quantity data CSV into a table named T_quantity using the defined options.
    T_quantity = readtable('Thesis Data - Architectural Quantity Cost.csv', opts_qty);
    % Confirms that the quantity data has been loaded successfully.
    fprintf('Architectural Quantity data loaded.\n');
    % Reads the unit cost data CSV into a table named T_unit_cost.
    T_unit_cost = readtable('Thesis Data - Achitectural Unit Cost.csv', opts_cost);
    % Confirms that the unit cost data has been loaded successfully.
    fprintf('Architectural Unit Cost data loaded.\n');
% Catches any error 'ME' that occurs within the 'try' block.
catch ME
    % Prints an error message, including the specific error details.
    fprintf('Error: %s\nMake sure both CSV files are in the correct directory.\n', ME.message);
    % Stops the script execution if files cannot be loaded.
    return;
end

% Calls a custom function 'clean_table' to preprocess the quantity table.
T_quantity_cleaned = clean_table(T_quantity);
% Calls the same custom function 'clean_table' to preprocess the unit cost table.
T_unit_cost_cleaned = clean_table(T_unit_cost);

% Applies the 'extract_budget' function to each row of the 'Year/Budget' column to get budget values.
budgets = rowfun(@extract_budget, T_quantity_cleaned(:, 'Year/Budget'), 'OutputFormat', 'uniform');
% Creates a new 'Budget' column in the quantity table with the extracted values.
T_quantity_cleaned.Budget = budgets;
% Removes the original 'Year/Budget' column as it's no longer needed.
T_quantity_cleaned.('Year/Budget') = [];
% Removes the 'Year/Budget' column from the unit cost table as well.
T_unit_cost_cleaned.('Year/Budget') = [];

% Merges the two cleaned tables into one table 'T_merged' based on a common 'Join_Key'.
T_merged = innerjoin(T_quantity_cleaned, T_unit_cost_cleaned, 'Keys', 'Join_Key');

% Filters the merged table to remove rows where the 'Budget' is NaN (Not-a-Number).
T_merged = T_merged(~isnan(T_merged.Budget), :);
% Filters the table further to keep only projects with a budget greater than 100,000.
T_merged = T_merged(T_merged.Budget > 100000, :);
% Prints a confirmation message with the final number of projects after merging and cleaning.
fprintf('Tables merged successfully using Join_Key. Working with %d common projects.\n', height(T_merged));


% =========================================================================
% Section 3: Granular Feature Engineering & Visualization
% =========================================================================
% Prints a header for the feature engineering section.
fprintf('\n--- Section 3: Engineering Granular Features ---\n');

% Initializes an empty cell array to store the names of newly created cost features.
individual_cost_features = {};
% Defines a list of base feature names that will be used to calculate estimated costs.
base_feature_cols = {
    'Quantity of plaster (sq.m.)', 'Quantity of glazed tiles (sq.m.)', ...
    'Painting masonry (sq.m.)', 'painting wood (sq.m.)', ...
    'painting metal (sq.m.)', 'Area of CHB 100mm (sq.m.)', ...
    'Area of CHB 150mm (sq.m.)'
};

% Starts a loop to iterate through each base feature name.
for i = 1:numel(base_feature_cols)
    % Gets the current feature name from the list.
    col = base_feature_cols{i};
    % Creates the corresponding column name for quantity (from the first table).
    qty_col = [col, '_T_quantity_cleaned'];
    % Creates the corresponding column name for unit cost (from the second table).
    cost_col = [col, '_T_unit_cost_cleaned'];
    
    % Creates a valid and descriptive name for the new feature column by cleaning the base name.
    new_cost_feature = regexprep(col, {' \(sq\.m\.\)', 'Quantity of ', 'Area of ', ' '}, {'', '', '', '_'});
    % Appends '_Est_Cost' to the new feature name.
    new_cost_feature = [new_cost_feature, '_Est_Cost'];

    % Checks if both the quantity and cost columns exist in the merged table.
    if ismember(qty_col, T_merged.Properties.VariableNames) && ismember(cost_col, T_merged.Properties.VariableNames)
        % If they exist, calculate the estimated cost by multiplying quantity by unit cost.
        T_merged.(new_cost_feature) = T_merged.(qty_col) .* T_merged.(cost_col);
    else
        % If columns are missing, create the new feature column and fill it with zeros.
        T_merged.(new_cost_feature) = zeros(height(T_merged),1);
        % Prints a warning that data was missing for this feature.
        fprintf('Warning: missing columns for %s; filled with zeros.\n', new_cost_feature);
    end
    % Adds the name of the newly created feature to the list.
    individual_cost_features{end+1} = new_cost_feature;
end
% Confirms that the new granular cost features have been created.
fprintf('Created new granular cost features.\n');

% Specifies the column containing the project description text.
project_description_col = 'Project_Name_T_quantity_cleaned';

% Uses regular expressions to find numbers followed by 'sty' (storey) in the project description.
storeys_cell = regexp(T_merged.(project_description_col), '(\d+)\s*(?:sty|x)', 'tokens', 'once');
% Gets the total number of rows in the table.
num_rows = height(T_merged);
% Initializes a column of NaNs to store the number of storeys.
num_storeys = NaN(num_rows, 1);
% Loops through each row to extract the number of storeys.
for i = 1:num_rows
    % Checks if a match was found for the current row.
    if ~isempty(storeys_cell{i})
        % Converts the extracted text (a number) into a numeric value.
        num_storeys(i) = str2double(storeys_cell{i}{1});
    end
end
% Creates a new 'Num_Storeys' column in the table.
T_merged.Num_Storeys = num_storeys;
% Replaces any remaining NaN values with the median number of storeys to fill missing data.
T_merged.Num_Storeys(isnan(T_merged.Num_Storeys)) = median(T_merged.Num_Storeys, 'omitnan');

% Uses regular expressions to find numbers followed by 'cl' (classroom).
classrooms_cell = regexp(T_merged.(project_description_col), '(\d+)\s*cl', 'tokens', 'once');
% Initializes a column of NaNs to store the number of classrooms.
num_classrooms = NaN(num_rows, 1);
% Loops through each row to extract the number of classrooms.
for i = 1:num_rows
    % Checks if a match was found for the current row.
    if ~isempty(classrooms_cell{i})
        % Converts the extracted text (a number) into a numeric value.
        num_classrooms(i) = str2double(classrooms_cell{i}{1});
    end
end
% Creates a new 'Num_Classrooms' column in the table.
T_merged.Num_Classrooms = num_classrooms;
% Replaces any remaining NaN values with the median number of classrooms.
T_merged.Num_Classrooms(isnan(T_merged.Num_Classrooms)) = median(T_merged.Num_Classrooms, 'omitnan');

% Creates a new column 'Budget_log' by applying a log transformation to the 'Budget'. This helps handle skewed data.
T_merged.Budget_log = log1p(T_merged.Budget);

% --- PLOTTING AND SAVING CORRELATION MATRIX ---
% Create a figure, but keep it hidden from view.
fig1 = figure('Visible', 'off'); 

% Defines the list of features to include in the correlation matrix.
heatmap_cols = [individual_cost_features, 'Num_Storeys', 'Num_Classrooms', 'Budget'];
% Calculates the pairwise correlation coefficient matrix for the selected columns.
correlation_matrix = corrcoef(T_merged{:, heatmap_cols}, 'Rows', 'pairwise');
% Generates a heatmap visualization of the correlation matrix.
h = heatmap(heatmap_cols, heatmap_cols, correlation_matrix);
% Sets the title of the heatmap.
h.Title = 'Correlation Matrix of Granular Features';
% Sets the color scheme for the heatmap.
h.Colormap = parula;

% Save the current figure (the heatmap) to a PNG file.
exportgraphics(fig1, 'correlation_matrix.png', 'Resolution', 300);
fprintf('Correlation heatmap saved as correlation_matrix.png\n');

% Close the hidden figure to free up memory.
close(fig1);


% =========================================================================
% Section 4: Data Preparation for the ANN Model
% =========================================================================
% Prints a header for the data preparation section.
fprintf('\n--- Section 4: Preparing Data for the ANN Model ---\n');

% Defines the final list of feature columns to be used as inputs for the model.
final_feature_columns = [individual_cost_features, 'Num_Storeys', 'Num_Classrooms'];
% Extracts the input features (X) from the table into a numeric matrix.
X = T_merged{:, final_feature_columns};     % `X` is a matrix where rows are projects and columns are features.
% Extracts the target variable (y), the log-transformed budget, into a column vector.
y = T_merged.Budget_log;                    % `y` is the value we want the model to predict.

% Prints the number of input features being used.
fprintf('Training model with %d input features.\n', size(X, 2));

% Sets the random number generator seed to 42 for reproducibility of results.
rng(42);
% Gets the total number of data points (projects).
num_data = size(X, 1);
% Creates a randomly shuffled list of indices from 1 to the total number of data points.
shuffled_indices = randperm(num_data);

% Calculates the index at which to split the data (80% for training).
split_point = floor(0.8 * num_data);
% Selects the first 80% of shuffled indices for the training set.
idxTrain = shuffled_indices(1:split_point);
% Selects the remaining 20% of shuffled indices for the testing set.
idxTest = shuffled_indices(split_point+1:end);

% Creates the training feature set using the training indices.
X_train = X(idxTrain, :);   % numTrain x numFeatures
% Creates the training target set using the training indices.
y_train = y(idxTrain, :);   % numTrain x 1
% Creates the testing feature set using the testing indices.
X_test = X(idxTest, :);
% Creates the testing target set using the testing indices.
y_test = y(idxTest, :);

% Standardizes the training features (z-score scaling) and stores the mean (mu) and standard deviation (sigma).
[X_train_scaled, scaler_X_mu, scaler_X_sigma] = zscore(X_train);

% Checks if any feature has a standard deviation of zero.
zeroSigmaIdx = scaler_X_sigma == 0;
% If any feature has zero standard deviation...
if any(zeroSigmaIdx)
    % ...set its sigma to 1 to prevent division by zero during scaling.
    scaler_X_sigma(zeroSigmaIdx) = 1; 
    % Display a warning message.
    warning('Some features had zero standard deviation; sigma set to 1 for those features.');
end

% Applies the same scaling (using the mean and sigma from the training set) to the test features.
X_test_scaled = (X_test - scaler_X_mu) ./ scaler_X_sigma;

% Finds the minimum value of the training target variable.
scaler_y_min = min(y_train);
% Finds the maximum value of the training target variable.
scaler_y_max = max(y_train);
% Calculates the range of the training target variable.
y_range = scaler_y_max - scaler_y_min;
% Checks if the range is zero (i.e., all target values are the same).
if y_range == 0
    % If so, the scaled targets will all be zero.
    y_train_scaled = zeros(size(y_train));
    % Displays a warning as the model cannot learn from constant data.
    warning('All training targets are identical. y_train_scaled set to zeros.');
else
    % Otherwise, scales the training targets to a range (min-max scaling).
    y_train_scaled = (y_train - scaler_y_min) / y_range;
end

% Ensures the scaled training target is a column vector (N-by-1).
y_train_scaled = y_train_scaled(:);

% Confirms that the data splitting and scaling process is complete.
fprintf('Data has been split and scaled.\n');


% =========================================================================
% Section 5: Build and Train the Artificial Neural Network
% =========================================================================
% Prints a header for the ANN building and training section.
fprintf('\n--- Section 5: Building and Training the ANN ---\n');

% Gets the number of input features, which defines the size of the input layer.
input_size = size(X_train_scaled, 2);

% Defines the architecture of the neural network layer by layer.
layers = [
    % Input layer: Specifies the number of input features.
    featureInputLayer(input_size, 'Name', 'input')
    % First hidden layer: A fully connected layer with 128 neurons.
    fullyConnectedLayer(128, 'Name', 'fc1')
    % Activation function: ReLU introduces non-linearity.
    reluLayer('Name', 'relu1')
    % Dropout layer: Randomly sets 30% of inputs to zero to prevent overfitting.
    dropoutLayer(0.3, 'Name', 'dropout1')
    % Second hidden layer: A fully connected layer with 64 neurons.
    fullyConnectedLayer(64, 'Name', 'fc2')
    % Activation function for the second hidden layer.
    reluLayer('Name', 'relu2')
    % Dropout layer for the second hidden layer (20% dropout).
    dropoutLayer(0.2, 'Name', 'dropout2')
    % Third hidden layer: A fully connected layer with 32 neurons.
    fullyConnectedLayer(32, 'Name', 'fc3')
    % Activation function for the third hidden layer.
    reluLayer('Name', 'relu3')
    % Output layer: A fully connected layer with 1 neuron for the single predicted value.
    fullyConnectedLayer(1, 'Name', 'output')
    % Regression layer: Specifies the loss function for a regression problem (e.g., mean squared error).
    regressionLayer('Name', 'regression')
];

% Defines the training options for the network.
options = trainingOptions('adam', ...          % Optimizer algorithm.
    'InitialLearnRate', 0.001, ...          % The starting learning rate.
    'MaxEpochs', 200, ...                   % The maximum number of training cycles.
    'MiniBatchSize', 16, ...                % The number of samples to use in each training iteration.
    'Shuffle', 'every-epoch', ...           % Shuffles the data before each epoch to improve training.
    'Plots', 'none', ...                    % Disables the interactive training plot window.
    'Verbose', false);                      % Prevents detailed text output during training.

% Converts data to single precision floating-point, which can speed up training.
X_train_scaled = single(X_train_scaled);
X_test_scaled  = single(X_test_scaled);
y_train_scaled = single(y_train_scaled);

% This is a crucial note: trainNetwork expects data where rows are observations. No transpose is needed.
% *** CHANGED: Capture the training history in the 'trainInfo' variable.
[net, trainInfo] = trainNetwork(X_train_scaled, y_train_scaled, layers, options);
% Confirms that the training process is complete.
fprintf('Neural Network training complete.\n');

% --- NEW: PLOTTING AND SAVING TRAINING PROGRESS GRAPH ---
% Create a hidden figure to plot the training progress.
fig_train = figure('Visible', 'off');

% Create the top subplot for RMSE.
subplot(2, 1, 1);
plot(trainInfo.TrainingRMSE);
title('Training Progress (RMSE)');
ylabel('RMSE');
xlabel('Iteration');
grid on;

% Create the bottom subplot for Loss.
subplot(2, 1, 2);
plot(trainInfo.TrainingLoss);
title('Training Progress (Loss)');
ylabel('Loss');
xlabel('Iteration');
grid on;

% Save the entire figure with both subplots to a PNG file.
exportgraphics(fig_train, 'training_progress_graph.png', 'Resolution', 300);
fprintf('Training progress graph saved as training_progress_graph.png\n');

% Close the hidden figure.
close(fig_train);


% =========================================================================
% Section 6: Model Evaluation
% =========================================================================
% Prints a header for the model evaluation section.
fprintf('\n--- Section 6: Evaluating the ANN Model ---\n');

% Uses the trained network 'net' to make predictions on the scaled test data.
scaled_predictions = predict(net, X_test_scaled);  % Output is a column vector of scaled predictions.

% Checks if the original target range was zero.
if y_range == 0
    % If so, the prediction is simply the minimum value (as no variation was learned).
    log_predictions = scaled_predictions * 0 + scaler_y_min;
else
    % Reverses the min-max scaling to transform predictions back to the log-transformed scale.
    log_predictions = scaled_predictions .* (scaler_y_max - scaler_y_min) + scaler_y_min;
end

% Reverses the log transformation (expm1 is equivalent to exp(x)-1) to get the final budget predictions.
final_predictions = expm1(log_predictions);
% Also reverses the log transformation on the actual test targets for comparison.
y_test_actual = expm1(y_test);

% Calculates the R-squared value, a measure of how well the model explains the variance in the data.
r2_ann = 1 - sum((y_test_actual - final_predictions).^2) / sum((y_test_actual - mean(y_test_actual)).^2);
% Calculates the Mean Absolute Error (MAE), the average absolute difference between actual and predicted values.
mae_ann = mean(abs(y_test_actual - final_predictions));
% Calculates the Root Mean Squared Error (RMSE), the square root of the average squared differences.
rmse_ann = sqrt(mean((y_test_actual - final_predictions).^2));

% Prints the final performance metrics.
fprintf('\n--- Final Model Performance ---\n');
fprintf('R-squared (R²): %.4f\n', r2_ann);
fprintf('Mean Absolute Error (MAE): ₱%,.2f\n', mae_ann);
fprintf('Root Mean Squared Error (RMSE): ₱%,.2f\n', rmse_ann);

% --- PLOTTING AND SAVING RESULTS PLOT ---
% Create a new hidden figure for the results plot.
fig2 = figure('Visible', 'off');

% Creates a scatter plot of actual budget vs. predicted budget.
scatter(y_test_actual, final_predictions, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
% Keeps the current plot active to add more elements.
hold on;
% Plots a red dashed line representing a perfect prediction (y=x).
p_line = plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--', 'LineWidth', 2);
% Releases the plot.
hold off;
% Sets the title of the plot.
title('Actual vs. Predicted Project Budget');
% Sets the label for the x-axis.
xlabel('Actual Budget (PHP)');
% Sets the label for the y-axis.
ylabel('Predicted Budget (PHP)');
% Adds a legend to the plot.
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');
% Adds a grid to the plot for better readability.
grid on;
% Gets the current axes object.
ax = gca;
% Prevents scientific notation on the x-axis for clearer labels.
ax.XAxis.Exponent = 0;
% Prevents scientific notation on the y-axis.
ax.YAxis.Exponent = 0;

% Save the scatter plot figure to a PNG file.
exportgraphics(fig2, 'actual_vs_predicted_budget.png', 'Resolution', 300);
fprintf('Actual vs. Predicted plot saved as actual_vs_predicted_budget.png\n');

% Close the hidden figure.
close(fig2);


% =========================================================================
% Section 7: Save Final Assets
% =========================================================================
% Prints a header for the saving section.
fprintf('\n--- Section 7: Saving Final Model and Scalers ---\n');

% Saves the trained neural network object 'net' to a .mat file.
save('ann_granular_model.mat', 'net');
% Confirms that the model has been saved.
fprintf("ANN model saved as 'ann_granular_model.mat'\n");

% Saves the scaler variables (means, std devs, min, max) to a .mat file for later use.
save('scalers_granular.mat', 'scaler_X_mu', 'scaler_X_sigma', 'scaler_y_min', 'scaler_y_max');
% Confirms that the scalers have been saved.
fprintf("Feature and target scalers saved as 'scalers_granular.mat'.\n");

% Prints a final message indicating the script has completed successfully.
fprintf('\nProcess finished successfully.\n');