%% fatigue_estimate_RfRr
% recoverable fatigue increases in proportion to effort level after effort 
% decreases after rest
% no unrecoverable fatigue
function [ERR,Fat,Rfat,Ufat] = fatigue_estimate_RfRr(params, E, ratings, baseline)


%% Free parameters
alfa = params(1); %recovery rate
beta = params(2); %recoverable fatigue rate 

%% Empy arrays for fatigue scores
Rfat = NaN(length(E), 1); 
Ufat = repmat(0, length(E),1); %No unrecoverable fatigue
Fat = NaN(length(E), 1);

%%
% First trial:
for t=1
    % if exerted effort:
    if E(t) > 0
        % recoverable fatigue increases by beta parameter modulated by effort level
        Rfat(t) = baseline + (beta * E(t));
    %if rested
    elseif E(t) == 0
        % recoverable fatigue decreases by alfa parameter 
        Rfat(t) = baseline - (alfa * 1);
    end

    % Limit Rfat from going below the initial value
    if Rfat(t) < baseline
        Rfat(t) = baseline;
    else
        Rfat(t) = Rfat(t);
    end
    % Fatigue rating estimate = recoverable fatigue 
    Fat(t) = Rfat(t) + Ufat(t);
end

%Further trials
for t=2:length(E)
    % effort
    if E(t) > 0 
        Rfat(t) = Rfat(t-1) + (beta*E(t));
    % rest
    elseif E(t) == 0
        Rfat(t) = Rfat(t-1) - (alfa*1);
    end
    
    % Limit Rfat from going below the initial value
    if Rfat(t) < baseline
        Rfat(t) = baseline;
    else
        Rfat(t) = Rfat(t);
    end
    % Fatigue rating estimate = recoverable fatigue value
    Fat(t) = Rfat(t) + Ufat(t);
end
%% Fit measures
residual_squares = ((ratings - Fat).^2);
ERR = sum(residual_squares); %Error defined as Residual Sum of Squares
end