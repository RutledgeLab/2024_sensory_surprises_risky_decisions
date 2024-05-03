function result = fitmodel_omnibus_persevrb(indata,parameters_to_split)
    
    % features 6 parameters: mu, lambda, alphagain, alphaloss, persev,
    % risky bias
    % parameters_to_split is an argument that states which parameter to fit
    % on rare and common trials separately.

    % This script uses Maximum Likelihood Estimation (MLE) for parameter
    % estimation.
    
    options = optimset('Display','off','MaxIter',100000,'TolFun',1e-10,'TolX',1e-10,...
        'DiffMaxChange',1e-2,'DiffMinChange',1e-4,'MaxFunEvals',10000,'LargeScale','off');
    warning off; %display,iter to see outputs
    
    %% parameter inx from omnibus model, split fits
    

    betalabel_comm = {'\mu','\lambda','\alpha+','\alpha-','persev','risky bias'};
    inx_comm = [    1  1.1   0.8   0.8   0    0];
    lb_comm  = [ 0.01  0.5   0.3   0.3  -3   -3];
    ub_comm  = [   20    5   1.3   1.3   3    3];
    
    betalabel_rare = {'{\Delta} Mu','{\Delta} Lambda','{\Delta} Alpha+','{\Delta} Alpha-','{\Delta} persev','{\Delta} risky bias'}; 

    inx_rare = zeros(1,length(betalabel_rare));
    lb_rare  = [-20 -3 -1 -1 -3 -3];
    ub_rare  = [ 20  3  1  1  3  3];

    [splits,~] = ismember(betalabel_comm,parameters_to_split); % for the difference parameters that do not get fit, they are fixed to zero

    lb_rare(~splits) = 0;
    ub_rare(~splits) = 0;
    

    betalabel = [betalabel_comm betalabel_rare];
    inx = [inx_comm inx_rare]; 
    lb = [lb_comm lb_rare];
    ub = [ub_comm ub_rare];

  
    
    %% Setting up param + dParam inequalities --------------------------
    if sum(splits)>0 % if any splits exist
   
        idx_splitparam = find(splits); % get index of split parameter
        
        A = nan(numel(idx_splitparam)*2,size(betalabel,2)); % A (M by N matrix, where M is the number of inequalities and N is the number of variables in x)
        B = nan(numel(idx_splitparam)*2,1); % B (M by 1 matrix, where M is the number of inequalities)

        for p = 1:numel(idx_splitparam)
            
            newrow_ub = zeros(1,numel(betalabel_comm));
            newrow_lb = zeros(1,numel(betalabel_comm));
            newrow_ub(idx_splitparam(p)) = 1;
            newrow_lb(idx_splitparam(p)) = -1;


            A(2*p-1,:) = repmat(newrow_ub,1,2); % 1st row, for upper bound inequality 
            A(2*p,:)= repmat(newrow_lb,1,2); % 2nd row, for lower bound inequality
            
            B(2*p-1,:) = ub_comm(idx_splitparam(p));
            B(2*p,:) = -lb_comm(idx_splitparam(p));
            
        end
    else
        A = [];
        B = [];
        
    end

    %% set up the results matrix
    result = struct;
    result.data = indata;
    result.inx = inx;
    result.lb = lb;
    result.ub = ub;
    result.options = options;
    result.nosplit = ~splits;
    result.betalabel_all = betalabel;
    result.param_count = sum([true(1,length(betalabel_comm)) splits]);
    result.paramsactive = [true(1,length(betalabel_comm)) splits];
    result.betalabel = betalabel(result.paramsactive);
    try
        [b, ~, exitflag, output, ~, ~, H] = fmincon(@model_param, inx,A,B,[],[],lb,ub,[], options, result);
        clear temp;
        [loglike, utildiff, logodds, probchoice] = model_param(b, result);
        result.b  = b(result.paramsactive);
        result.b_all = b;  
        result.H  = H;
        se      = transpose(sqrt(diag(inv(result.H))));
        result.se = se(:);
        result.modelLL    = -loglike;
        result.nullLL     = log(0.5)*length(probchoice);
        result.pseudoR2   = 1 - result.modelLL / result.nullLL;
        result.LRstat     = -2*(result.nullLL - result.modelLL);
        result.LRtestp    = chi2pdf(result.LRstat,length(b(result.paramsactive)));
        result.exitflag   = exitflag;
        result.output     = output;
        result.utildiff   = utildiff;
        result.logodds    = logodds;
        result.probchoice = probchoice;
        result.BIC = -2 * (-loglike) + (result.param_count.*log(length(probchoice)));
        result.AIC = -2 * (-loglike) + 2*(result.param_count);

    catch
         fprintf(1,'model fit failed\n');
    end
    

end

function [loglike, utildiff, logodds, probchoice] = model_param(x, data)
    
    data.mu_comm       = x(1);
    data.lambda_comm    = x(2);
    data.alphagain_comm = x(3);
    data.alphaloss_comm = x(4);
    data.persev_comm = x(5);
    data.bias_comm = x(6);
    

    oddball = data.data(:,16)~=1;
    
    
    param_vecs{1} = repelem(data.mu_comm,size(data.data,1))';
    param_vecs{2} = repelem(data.lambda_comm,size(data.data,1))';
    param_vecs{3} = repelem(data.alphagain_comm,size(data.data,1))';
    param_vecs{4} = repelem(data.alphaloss_comm,size(data.data,1))';
    param_vecs{5} = repelem(data.persev_comm,size(data.data,1))';
    param_vecs{6} = repelem(data.bias_comm,size(data.data,1))';
    
    
    for p = 1:length(data.nosplit)
        if data.nosplit(p)==0 % if the parameter has a dParameter, we implement it here
            param_vecs{p}(oddball) = x(length(param_vecs)+p)+x(p); 
        end
    end
    
    mu_vec = param_vecs{1};
    lambda_vec = param_vecs{2};
    alphagain_vec = param_vecs{3};
    alphaloss_vec = param_vecs{4};
    persev_vec = param_vecs{5};
    bias_vec = param_vecs{6};
    
    
    % calculate value differences for common trials
    utilcertain = (data.data(:,3)>0).*abs(data.data(:,3)).^alphagain_vec - ...
        (data.data(:,3)<0).*lambda_vec.*abs(data.data(:,3)).^alphaloss_vec;
    winutil       = data.data(:,4).^alphagain_vec;
    lossutil      = -lambda_vec.*(-data.data(:,5)).^alphaloss_vec;
    utilgamble    = 0.5*winutil+0.5*lossutil;
    utildiff      = utilgamble - utilcertain;
    logodds       = (mu_vec.*utildiff);
    % 1 = previous trial was gamble, -1 = previous trial was safe
    persevlogodds = (persev_vec .* [0; (2*(data.data(1:end-1,7)==1)-1)]);
    
    
    probchoice = (1 ./ (1+exp(-(logodds + persevlogodds + bias_vec))));
    
    choice        = data.data(:,7);

    probchoice(probchoice==0) = eps;   %to prevent fminunc crashing from log zero
    probchoice(probchoice==1) = 1-eps;
    
    % loglike = - (choice.*log(probchoice) + (1-choice).*log(1-probchoice));
    loglike = - (transpose(choice(:))*log(probchoice(:)) + transpose(1-choice(:))*log(1-probchoice(:)));
    loglike = sum(loglike); %need on number to minimize


end

