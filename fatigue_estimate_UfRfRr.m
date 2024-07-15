%% fatigue_estimate_UfRfRr
% recoverable fatigue increases as a function of effort after effort and decreases after rest
% unrecoverable fatigue increases as a function of effort after effort and doesn't change after rest
function [ERR,Fat,Rfat,Ufat] = fatigue_estimate_UfRfRr(params, E, ratings, baseline)

%% Free parameters
alfa = params(1); %recovery rate
beta = params(2); %recoverable fatigue rate
gamma = params(3); %unrecoverable fatigue rate

%% Empty arrays for fatigue scores
Rfat = NaN(length(E), 1);
Ufat = NaN(length(E), 1);
Fat = NaN(length(E), 1);

%% 
% First trial:
for t=1
    % if exerted effort:
    if E(t) > 0
        % recoverable fatigue increases by beta parameter modulated by effort level
        Rfat(t) = baseline + (beta*E(t));
        % unrecoverable fatigue increases by gamma parameter modulated by effort level
        Ufat(t) = (gamma*E(t));
    %if rest trial / not done effort:
    elseif E(t) == 0 
        % recoverable fatigue decreases by alfa parameter
        Rfat(t) = baseline - (alfa*1);
        % unrecoverable fatigue value stays the same 
        Ufat(t) = 0;
    end
    
    % Limit Rfat from going below the initial value
    if Rfat(t) < baseline
        Rfat(t) = baseline;
    else
        Rfat(t) = Rfat(t);
    end
    
    % Fatigue rating estimate = short and unrecoverable fatigue value
    Fat(t) = Rfat(t) + Ufat(t);
    
end

%Further trials
for t=2:length(E)
    % effort
    if E(t) > 0
        Rfat(t) = Rfat(t-1) + (beta*E(t));
        Ufat(t) = Ufat(t-1) + (gamma*E(t));
    % rest
    elseif E(t) == 0
        Rfat(t) = Rfat(t-1) - (alfa*1);
        Ufat(t) = Ufat(t-1);
    end
    
    % Limit Rfat from going below the initial value
    if Rfat(t) < baseline
        Rfat(t) = baseline;
    else
        Rfat(t) = Rfat(t);
    end
    % Fatigue rating estimate = short and unrecoverable fatigue value
    Fat(t) = Rfat(t) + Ufat(t);
   
end
%% Fit measures
residual_squares = ((ratings - Fat).^2);
ERR = sum(residual_squares, "omitnan"); %Error defined as Residual Sum of Squares

end