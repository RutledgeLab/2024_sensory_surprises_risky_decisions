function result = fitmodel_pt(indata)

% RESULT = fitmodel_loss_aversion(INDATA)
%
% INDATA is a matrix with at least 7 columns (col 3 certain amount, col 4
% win amount, col 5 loss amount, col 7 chose risky is 1, chose safe is 0)


result           = struct;
result.data      = indata;
result.betalabel = {'mu','lambda','alpha+','alpha-'}; 
result.inx       = [1     1 0.8 0.8];   %initial values for parameters
result.lb        = [0.01  0.5 0.3 0.3]; %min values possible for design matrix
result.ub        = [20      5 1.3 1.3];   %max values
result.options   = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off;                    %to see outputs use 'Display','iter'

try
    [b, ~, exitflag, output, ~, ~, H] = fmincon(@mymodel,result.inx,[],[],[],[],result.lb,result.ub,[],result.options,result);
    clear temp;
    [loglike, utildiff, logodds, probchoice] = mymodel(b, result);
    result.b          = b;      %parameter estimates
    result.se         = transpose(sqrt(diag(inv(H)))); %SEs for parameters from inverse of the Hessian
    result.modelLL    = -loglike;
    result.nullLL     = log(0.5)*length(probchoice);
    result.pseudoR2   = 1 - result.modelLL / result.nullLL;
    result.LRstat     = -2*(result.nullLL - result.modelLL);
    result.LRtestp    = chi2pdf(result.LRstat,length(b));
    result.exitflag   = exitflag;
    result.output     = output;
    result.utildiff   = utildiff;
    result.logodds    = logodds;
    result.probchoice = probchoice;
catch
    fprintf(1,'model fit failed\n');
end

end


function [loglike, utildiff, logodds, probchoice] = mymodel(x, data)

data.mu         = x(1);
data.lambda     = x(2);
data.alphaplus  = x(3);
data.alphaminus = x(4);

[loglike, utildiff, logodds, probchoice] = mypt_model(data);

end


function [loglike, utildiff, logodds, probchoice] = mypt_model(data)

%data.data is a matrix with at least 7 columns (col 3 certain amount, col 4
%win amount, col 5 loss amount, col 7 chose risky is 1, chose safe is 0)
%data.lambda and data.mu are loss aversion and inverse temperature
%parameters. function returns -loglikelihood and vectors for trial-by-trial
%utility difference, logodds, and probability of taking the risky option

utilcertain   = (data.data(:,3)>0).*abs(data.data(:,3)).^data.alphaplus - ...
                (data.data(:,3)<0).*data.lambda.*abs(data.data(:,3)).^data.alphaminus;
winutil       = data.data(:,4).^data.alphaplus;                   %utility for potential risky gain
lossutil      = -data.lambda*(-data.data(:,5)).^data.alphaminus; %utility for potential risky loss
utilgamble    = 0.5*winutil+0.5*lossutil;         %utility for risky option
utildiff      = utilgamble - utilcertain;         %utility difference between risky and safe options
logodds       = data.mu*utildiff;                 %convert to logodds using noise parameter
probchoice    = 1 ./ (1+exp(-logodds));           %prob of choosing gamble
choice        = data.data(:,7);                   %1 chose risky, 0 chose safe


probchoice(probchoice==0) = eps;                  %to prevent fmincon crashing from log zero
probchoice(probchoice==1) = 1-eps;
loglike       = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
loglike       = sum(loglike);                     %number to minimize

end