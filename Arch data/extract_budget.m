% =========================================================================
% extract_budget.m
%
% Helper function to extract a budget value from a string. It looks for
% a number with more than 4 digits to avoid matching the year.
% =========================================================================
function budget = extract_budget(textCell)
    text = textCell{1}; % Extract string from the input cell
    budget = NaN;       % Default return value if no budget is found
    
    if ischar(text) || isstring(text)
        % Find all number-like patterns (including commas)
        matches = regexp(text, '[\d,]+\.?\d*', 'match');
        
        for i = 1:numel(matches)
            match_str = matches{i};
            % Assume the budget is the first number found that is > 4 chars long
            if length(match_str) > 4
                budget = str2double(strrep(match_str, ',', ''));
                return; % Exit the function once the budget is found
            end
        end
    end
end