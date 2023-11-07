function result = fitmodel_omnibus_OBdiff_model(indata,splitvec)

% Risky Bias Perseveration Difference model: 
% 8 parameters: mu, lambda, alpha+, alpha-, persev, delta_persev,
% riskybias, delta_riskybias

% Splitvec (length: 16 logical array) determines which parameters to split
% over rare and common in the Omnibus model.
% Depending on the splitvec input argument, this model can have up to 16
% parameters (difference parameters for all 8 parameters)

options = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off; %display,iter to see outputs
%eps should work but for some reason if the min is eps, it will still try
%negative numbers and the LL will be NaN and it will not converge

%% parameter inx from omnibus model, split fits

inx = [  1     0  1.1  0  0.8     0    0.8     0    0         0      0      0  0  0  0  0];
lb =  [0.01  -20  0.5 -3  0.3    -0.5  0.3    -0.5   -0.8    -0.5   -0.8   -0.5 -3 -6 -3 -6]; %min for data set
ub =  [20     20    5  3  1.3     0.5  1.3     0.5    0.8     0.5    0.8    0.5  3  6  3  6]; %mu/lambda/alphagain/alphaloss
betalabel = {'\mu','\Delta\mu','\lambda','\Delta\lambda','\alpha+','\Delta\alpha+','\alpha-','\Delta\alpha-',...
    '\beta+','\Delta\beta+','\beta-','\Delta\beta-','persev','{\Delta} persev','risky bias','{\Delta} risky bias'}; 
    
% instructions for which parameters to combine and which to split
inx = inx.*splitvec;
lb = lb.*splitvec;
ub = ub.*splitvec;
param_count = sum(splitvec);


dof = length(inx);
result = struct;
result.data = indata;
result.inx = inx;
result.lb = lb;
result.ub = ub;
result.options = options;
result.betalabel = betalabel;
result.param_count = param_count;

try
    [b, ~, exitflag, output, ~, ~, H] = fmincon(@model_param, inx, [],[],[],[],lb,ub,[], options, result);
    clear temp;
    [loglike, utildiff, logodds, probchoice] = model_param(b, result);
    result.b  = b;
    result.b_tot = b(splitvec);
    result.H  = H;
    se      = transpose(sqrt(diag(inv(result.H))));
    result.se = se(:);
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
    result.BIC = -2 * (-loglike) + (sum(splitvec)*log(length(probchoice)));
    result.AIC = -2 * (-loglike) + 2*(sum(splitvec));
catch
    fprintf(1,'model fit failed\n');
    %lasterr
end


function [loglike, utildiff, logodds, probchoice] = model_param(x, data)

data.mu_comm       = x(1);
data.dMu        = x(2);
data.lambda_comm    = x(3);
data.dLambda    = x(4);
data.alphagain_comm = x(5);
data.dAlphagain = x(6);
data.alphaloss_comm = x(7);
data.dAlphaloss = x(8);
data.betagain_comm = x(9);
data.dBetagain = x(10);
data.betaloss_comm = x(11);
data.dBetaloss = x(12);
data.persev_comm = x(13);
data.dPersev = x(14);
data.bias_comm = x(15);
data.dBias = x(16);


[loglike, utildiff, logodds, probchoice] = apav_model_OBdiff_spliteverything(data);

function [loglike, utildiff, logodds, probchoice] = apav_model_OBdiff_spliteverything(data)


common = data.data(:,16)==1;
oddball = data.data(:,16)~=1;

% calculate value differences for common trials
utilcertain = (data.data(:,3)>0).*abs(data.data(:,3)).^data.alphagain_comm - ...
    (data.data(:,3)<0).*data.lambda_comm.*abs(data.data(:,3)).^data.alphaloss_comm;
winutil       = data.data(:,4).^data.alphagain_comm;
lossutil      = -data.lambda_comm*(-data.data(:,5)).^data.alphaloss_comm;
utilgamble    = 0.5*winutil+0.5*lossutil;
utildiff      = utilgamble - utilcertain;
logodds       = (data.mu_comm*utildiff);

% 1 = previous trial was gamble, -1 = previous trial was safe
persevlogodds = (data.persev_comm * [0; (2*(data.data(1:end-1,7)==1)-1)]);
oddball_persev_temp = (data.persev_comm + data.dPersev) * [0; (2*(data.data(1:end-1,7)==1)-1)];
persevlogodds(oddball) = oddball_persev_temp(oddball);

bias_term = repelem(data.bias_comm,length(data.data))';
bias_term_temp = repelem((data.bias_comm + data.dBias),length(data.data))';
bias_term(oddball) = bias_term_temp(oddball);

% do the same for oddball trials
utilcertain = (data.data(:,3)>0).*abs(data.data(:,3)).^(data.alphagain_comm + data.dAlphagain) - ...
     (data.data(:,3)<0).*(data.lambda_comm+data.dLambda).*abs(data.data(:,3)).^(data.alphaloss_comm + data.dAlphaloss);
winutil       = data.data(:,4).^(data.alphagain_comm + data.dAlphagain);
lossutil      = (-(data.lambda_comm + data.dLambda))*(-data.data(:,5)).^(data.alphaloss_comm + data.dAlphaloss);
utilgamble    = 0.5*winutil+0.5*lossutil;
utildiff_ob      = utilgamble - utilcertain;

logodds_temp    = (data.mu_comm + data.dMu)*utildiff_ob;
logodds(oddball) = logodds_temp(oddball);

probchoice = (1 ./ (1+exp(-(logodds + persevlogodds + bias_term))));

betagain_term = repelem(data.betagain_comm,length(data.data))';
betagain_term_temp = repelem((data.betagain_comm + data.dBetagain),length(data.data))';
betagain_term(oddball) = betagain_term_temp(oddball);

betaloss_term = repelem(data.betaloss_comm,length(data.data))';
betaloss_term_temp = repelem((data.betaloss_comm + data.dBetaloss),length(data.data))';
betaloss_term(oddball) = betaloss_term_temp(oddball);

probchoice(data.data(:,3)>0) = (probchoice(data.data(:,3)>0).*(1-abs(betagain_term(data.data(:,3)>0)))) + max([betagain_term(data.data(:,3)>0),zeros(sum(data.data(:,3)>0),1)],[],2);
probchoice(data.data(:,3)<0) = (probchoice(data.data(:,3)<0).*(1-abs(betaloss_term(data.data(:,3)<0)))) + max([betaloss_term(data.data(:,3)<0),zeros(sum(data.data(:,3)<0),1)],[],2);


choice        = data.data(:,7);

probchoice(probchoice==0) = eps;   %to prevent fminunc crashing from log zero
probchoice(probchoice==1) = 1-eps;

loglike = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
loglike = sum(loglike); %need on number to minimize




