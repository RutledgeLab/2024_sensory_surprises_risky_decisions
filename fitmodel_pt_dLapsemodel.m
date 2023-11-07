function result = fitmodel_pt_dLapsemodel(indata)


options = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
    'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
warning off; %display,iter to see outputs
%eps should work but for some reason if the min is eps, it will still try
%negative numbers and the LL will be NaN and it will not converge

inx = [1    1.1  0.8   0.8   0      0];
lb =  [0.01 0.5  0.3   0.3   0   -0.3]; %min for data set
ub =  [20     5  1.3   1.3   0.9  0.3]; %mu/lambda/alphagain/alphaloss
betalabel = {'mu','lambda','alpha+','alpha-','lapse-comm','dLapse'}; 

dof = length(inx);
result = struct;
result.data = indata;
result.inx = inx;
result.lb = lb;
result.ub = ub;
result.options = options;
result.betalabel = betalabel;

try
    [b, ~, exitflag, output, ~, ~, H] = fmincon(@model_param, inx, [],[],[],[],lb,ub,[], options, result);
    clear temp;
    [loglike, utildiff, logodds, probchoice] = model_param(b, result);
    result.b  = b;
    result.H  = H;
    se      = transpose(sqrt(diag(inv(result.H))));
    result.se = se(:);
    result.modelLL    = -loglike;
    result.nullLL     = log(0.5)*length(probchoice);
    result.pseudoR2   = 1 - (result.modelLL / result.nullLL);
    result.LRstat     = -2*(result.nullLL - result.modelLL);
    result.LRtestp    = chi2pdf(result.LRstat,length(b));
    result.exitflag   = exitflag;
    result.output     = output;
    result.utildiff   = utildiff;
    result.logodds    = logodds;
    result.probchoice = probchoice;
catch
    fprintf(1,'model fit failed\n');
    %lasterr
end


function [loglike, utildiff, logodds, probchoice] = model_param(x, data)

data.mu        = x(1);
data.lambda    = x(2);
data.alphagain = x(3);
data.alphaloss = x(4);
data.lapse_comm = x(5);
data.dLapse = x(6);


[loglike, utildiff, logodds, probchoice] = pt_model_2lapsemodel(data);

function [loglike, utildiff, logodds, probchoice] = pt_model_2lapsemodel(data)


common = data.data(:,16)==1;
oddball = data.data(:,16)~=1;

utilcertain = (data.data(:,3)>0).*abs(data.data(:,3)).^data.alphagain - ...
    (data.data(:,3)<0).*data.lambda.*abs(data.data(:,3)).^data.alphaloss;
winutil       = data.data(:,4).^data.alphagain;
lossutil      = -data.lambda*(-data.data(:,5)).^data.alphaloss;
utilgamble    = 0.5*winutil+0.5*lossutil;
utildiff      = utilgamble - utilcertain;
logodds       = data.mu*utildiff;

probchoice = 1 ./ (1+exp(-logodds));
probchoice = (probchoice.*(1-data.lapse_comm))+ (data.lapse_comm/2);
probchoice_odd = (probchoice.*(1-(data.lapse_comm + data.dLapse)))+ (data.lapse_comm + data.dLapse)/2;
probchoice(oddball) = probchoice_odd(oddball);

choice        = data.data(:,7);

%change the model so that prob can either be (0,x) or (x,1) but not (x,y)

probchoice(probchoice==0) = eps;   %to prevent fminunc crashing from log zero
probchoice(probchoice==1) = 1-eps;
loglike = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
loglike = sum(loglike); %need on number to minimize