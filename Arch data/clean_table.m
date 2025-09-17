% =========================================================================
% clean_table.m
%
% Helper function to clean the raw data tables. This function is called
% from the main script.
% =========================================================================
function T = clean_table(T)
    % --- Step 1: Reliably Rename the First Column ---
    if ismember('Var1', T.Properties.VariableNames)
        T = renamevars(T, 'Var1', 'Project_Name');
    else
        fprintf("Warning: First column was not named 'Var1'. Renaming '%s' to 'Project_Name'.\n", T.Properties.VariableNames{1});
        T = renamevars(T, T.Properties.VariableNames{1}, 'Project_Name');
    end

    % ========================================================================
    %  *** NEW FEATURE: Standardize '2x4' format to '2 sty 4 cl' ***
    %  This is done before creating the join key or extracting features
    %  to ensure consistency across all project names.
    % ========================================================================
    num_rows = height(T);
    for i = 1:num_rows
        name = T.Project_Name{i};
        % Pattern: (a number), optional space, 'x' or 'X', optional space, (a number)
        pattern = '(\d+)\s*[xX]\s*(\d+)';
        % Replacement: the first number, ' sty ', the second number, ' cl'
        replacement = '$1 sty $2 cl';
        T.Project_Name{i} = regexprep(name, pattern, replacement);
    end
    
    % --- Step 3: Create a Reliable Join Key ---
    join_keys = cell(num_rows, 1);
    pattern = '(?i)(project\s*\d+)'; 
    
    for i = 1:num_rows
        name = T.Project_Name{i};
        tokens = regexp(name, pattern, 'tokens', 'once');
        if ~isempty(tokens)
            key = lower(tokens{1});
            key = strrep(key, ' ', '');
            join_keys{i} = key;
        else
            join_keys{i} = lower(strtrim(name));
        end
    end
    T.Join_Key = join_keys;

    % --- Step 4: Clean Feature Columns ---
    if ismember('Architectural aspect', T.Properties.VariableNames)
        T.('Architectural aspect') = [];
    end
    
    feature_cols = {
        'Quantity of plaster (sq.m.)', 'Quantity of glazed tiles (sq.m.)', ...
        'Painting masonry (sq.m.)', 'painting wood (sq.m.)', ...
        'painting metal (sq.m.)', 'Area of CHB 100mm (sq.m.)', ...
        'Area of CHB 150mm (sq.m.)'
    };
    
    for i = 1:numel(feature_cols)
        col = feature_cols{i};
        if ismember(col, T.Properties.VariableNames)
            temp_col = string(T.(col));
            temp_col = strrep(temp_col, ',', '');
            temp_col = strrep(temp_col, '`', '');
            temp_col = strrep(temp_col, '-', 'NaN');
            T.(col) = str2double(temp_col);
        end
    end
    
    % --- Step 5: Impute Missing Values ---
    for i = 1:width(T)
        if isnumeric(T{:, i})
            col_data = T{:, i};
            if any(isnan(col_data))
                median_val = median(col_data, 'omitnan');
                col_data(isnan(col_data)) = median_val;
                T{:, i} = col_data;
            end
        end
    end
end