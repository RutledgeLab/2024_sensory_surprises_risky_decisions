%% Surprising Sounds bias risky decision making

% load the seven studies
clear;
tic

setFigureDefaults;

load('Exp1_data.mat')
exp1 = alldata;

load('Exp2_data.mat')
exp2 = alldata;

load('Exp3_data.mat')
exp3 = alldata;

load('Exp4_data.mat')
exp4 = alldata; 

load('Exp5_data.mat')
exp5 = alldata;

load('Exp6_data.mat')
exp6 = alldata;

load('Exp7_data.mat')
exp7 = alldata;


%% Combine individual experiments

datasets = {exp1,exp2};
exp1exp2 = vertcat(datasets{:}); % combines multiple struct arrays

datasets = {exp3,exp4};
exp3exp4 = vertcat(datasets{:}); % combines multiple struct arrays

datasets = {exp5,exp6};
exp5exp6 = vertcat(datasets{:}); % combines multiple struct arrays

%% select a single study to make plots for

alldata = exp1exp2;
study_title = 'Experiment 1 & 2';

%% select 2 individual studies I want to compare


study1 = exp1;
study_title_1 = 'Exp. 1';

study2 = exp2;
study_title_2 = 'Exp. 2';


%%%%%%%%%%%%% MODEL FREE PLOTS %%%%%%%%%%%%%%%%%%%%%

%% Figure 2A: Average Gambling,common vs rare trials, Study 1 vs study 2

gam_oddballs = nan(length(study1),2);
gam_common = nan(length(study1),2);
gam_avg_study1 = nan(length(study1),1);
gam_avg_study2 = nan(length(study1),1);

for s = 1:length(study1)
    t = study1(s).data;
    % isolating trials where gamble is chosen
    common = t(:,16)==1; oddball = t(:,16)~=1;
    gam_common(s,1) = mean(t(common,7),'omitnan')*100; % common study 1
    gam_oddballs(s,1) = mean(t(oddball,7),'omitnan')*100; % oddball study 1  
    gam_avg_study1(s,1) = mean(t(:,7),'omitnan')*100;
end

for s = 1:length(study2)
    t = study2(s).data;
    % isolating trials where gamble is chosen
    common = t(:,16)==1; oddball = t(:,16)~=1;
    gam_common(s,2) = mean(t(common,7),'omitnan')*100; % common mixed study 2
    gam_oddballs(s,2) = mean(t(oddball,7),'omitnan')*100; % oddball study 2
    gam_avg_study2(s,1) = mean(t(:,7),'omitnan')*100;
end

list_yloc = [60,56];
ylabel_text = 'Chose risky option (%)';
behav_measure = 'risk taking';
modelfree_avg_fourbarplot(gam_common,gam_oddballs,gam_avg_study1,gam_avg_study2,list_yloc,ylabel_text,study_title_1,study_title_2,behav_measure);

% save the source data
gam_common_study1 = gam_common(:,1);
gam_rare_study1 = gam_oddballs(:,1);
gam_overall_study1 = gam_avg_study1;

gam_common_study2 = gam_common(:,2);
gam_rare_study2 = gam_oddballs(:,2);
gam_overall_study2 = gam_avg_study2;

source_data_2a = table(gam_common_study1,gam_rare_study1,gam_overall_study1,gam_common_study2,gam_rare_study2,gam_overall_study2);
% writetable(source_data_2a,'source_data_2a.csv')

%% Figure 2B: Average Gambling 6 bar plot
gambling_frequency_oddballs = zeros(length(alldata),3);
gambling_frequency_common = zeros(length(alldata),3);

for s = 1:length(alldata)
    t = alldata(s).data;
    % isolating trials where gamble is chosen
    gain_d = t(:,3)>0; mixed_d = t(:,3)==0; loss_d = t(:,3)<0;
    common = t(:,16)==1; 
    oddball = t(:,16)~=1;
    gambling_frequency_common(s,1) = mean(t(common & gain_d,7),'omitnan')*100; % common gain domain
    gambling_frequency_oddballs(s,1) = mean(t(oddball & gain_d,7),'omitnan')*100; % oddball gain   
    gambling_frequency_common(s,2) = mean(t(common & mixed_d,7),'omitnan')*100; % common mixed domain
    gambling_frequency_oddballs(s,2) = mean(t(oddball & mixed_d,7),'omitnan')*100; % oddball mixed
    gambling_frequency_common(s,3) = mean(t(common & loss_d,7),'omitnan')*100; % common loss domain
    gambling_frequency_oddballs(s,3) = mean(t(oddball & loss_d,7),'omitnan')*100; % oddball loss
   
end


list_yloc = [75,65,45];
ylabel_text= 'Chose risky option (%)';
sixbarplot(gambling_frequency_common,gambling_frequency_oddballs,list_yloc,ylabel_text,study_title);

% save the source data
gain_common = gambling_frequency_common(:,1);
mixed_common = gambling_frequency_common(:,2);
loss_common = gambling_frequency_common(:,3);

gain_rare = gambling_frequency_oddballs(:,1);
mixed_rare = gambling_frequency_oddballs(:,2);
loss_rare = gambling_frequency_oddballs(:,3);

source_data_2b = table(gain_common,mixed_common,loss_common,gain_rare,mixed_rare,loss_rare);
% writetable(source_data_2b,'source_data_2b.csv')

%% Figure 3A: Average Stay common vs rare trials, Study 1 vs study 2

gam_stay_oddballs = zeros(length(study1),2);
gam_stay_common = zeros(length(study1),2);
stay_rate_avg_study1 = zeros(length(study1),1);
stay_rate_avg_study2 = zeros(length(study1),1);

for s = 1:length(study1)
    t = study1(s).data;
    stayed = diff(t(:,7))==0; %repeated choice to stimulus
    % isolating trials where gamble is chosen
    common = t(2:end,16)==1; oddball = t(2:end,16)~=1;
    gam_stay_common(s,1) = mean(stayed(common),'omitnan')*100; % common study 1
    gam_stay_oddballs(s,1) = mean(stayed(oddball),'omitnan')*100; % oddball study 1  
    stay_rate_avg_study1(s,1) = mean(stayed,'omitnan')*100;
end

for s = 1:length(study2)
    t = study2(s).data;
    stayed = diff(t(:,7))==0; %repeated choice to stimulus
    % isolating trials where gamble is chosen
    common = t(2:end,16)==1; oddball = t(2:end,16)~=1; 
    gam_stay_common(s,2) = mean(stayed(common),'omitnan')*100; % common mixed study 2
    gam_stay_oddballs(s,2) = mean(stayed(oddball),'omitnan')*100; % oddball study 2
    stay_rate_avg_study2(s,1) = mean(stayed,'omitnan')*100;
end


list_yloc = [65,62];
ylabel_text = 'Stayed with previous choice (%)';
behav_measure = 'stay';
modelfree_avg_fourbarplot(gam_stay_common,gam_stay_oddballs,stay_rate_avg_study1,stay_rate_avg_study2,list_yloc,ylabel_text,study_title_1,study_title_2,behav_measure);

% save the source data
stay_common_study1 = gam_stay_common(:,1);
stay_rare_study1 = gam_stay_oddballs(:,1);
stay_overall_study1 = stay_rate_avg_study1;

stay_common_study2 = gam_stay_common(:,2);
stay_rare_study2 = gam_stay_oddballs(:,2);
stay_overall_study2 = stay_rate_avg_study2;

source_data_3a = table(stay_common_study1,stay_rare_study1,stay_overall_study1,stay_common_study2,stay_rare_study2,stay_overall_study2);
% writetable(source_data_3a,'source_data_3a.csv')


%%  Figure 3B: Average Stay 6 bar plot

gam_stay_oddballs = zeros(length(alldata),3);
gam_stay_common = zeros(length(alldata),3);

for s = 1:length(alldata)

    t = alldata(s).data;
    stayed = diff(t(:,7))==0; %repeated choice to stimulus
    % isolating trials where gamble is chosen
    gain_d = t(2:end,3)>0; mixed_d = t(2:end,3)==0; loss_d = t(2:end,3)<0;
    common = t(2:end,16)==1; 
    oddball = t(2:end,16)~=1;

    gam_stay_common(s,1) = mean(stayed(common & gain_d))*100; % common gain domain
    gam_stay_oddballs(s,1) = mean(stayed(oddball & gain_d))*100; % oddball gain   
    gam_stay_common(s,2) = mean(stayed(common & mixed_d))*100; % common mixed domain
    gam_stay_oddballs(s,2) = mean(stayed(oddball & mixed_d))*100; % oddball mixed
    gam_stay_common(s,3) = mean(stayed(common & loss_d))*100; % common loss domain
    gam_stay_oddballs(s,3) = mean(stayed(oddball & loss_d))*100; % oddball loss

end

list_yloc = [62,62,62];
ylabel_text = 'Stayed with previous choice (%)';
sixbarplot(gam_stay_common,gam_stay_oddballs,list_yloc,ylabel_text,study_title);

% save the source data
gain_common = gam_stay_common(:,1);
mixed_common = gam_stay_common(:,2);
loss_common = gam_stay_common(:,3);

gain_rare = gam_stay_oddballs(:,1);
mixed_rare = gam_stay_oddballs(:,2);
loss_rare = gam_stay_oddballs(:,3);

source_data_3b = table(gain_common,mixed_common,loss_common,gain_rare,mixed_rare,loss_rare);
% writetable(source_data_3b,'source_data_3b.csv')

%%%%%%%% MODEL BASED RESULTS %%%%%%%%%%%%%%%%
%% singling out a set of parameter splits


model_name = 'Risky Bias Perseveration Difference model';
betalabel = {'\mu','\lambda','\alpha+','\alpha-','persev','risky bias'}; 
parameters_to_split = {'persev','risky bias'};
numParams = length(horzcat(betalabel,parameters_to_split));


betalabel_rb = {'\mu','\lambda','\alpha+','\alpha-','risky bias'}; 
parameters_to_split_rb = {'risky bias'};
numParams_rb = length(horzcat(betalabel_rb,parameters_to_split_rb));


apav_params = nan(length(alldata),numParams);
apav_params_study1 = nan(length(study1),numParams);
apav_params_study2 = nan(length(study2),numParams);
pseudor2_study1 = nan(length(study1),1);
pseudor2_study2 = nan(length(study2),1);
pseudor2_study12 = nan(length(alldata),1);

pt_probstay_idx = 19;
model1_risktaking_idx = 20;
model2_risktaking_idx = 21;
fullmodel_risktaking_idx = 22;
model1_stay_idx =  24;
model2_stay_idx = 25;
fullmodel_stay_idx = 23;


for s=1:length(alldata)
    fprintf(sprintf('fitting model to participant %.0f of %.0f...\n',s,length(alldata)))
    result = fitmodel_pt(alldata(s).data); %fits 4-parameter Prospect Theory model
    alldata(s).result_pt = result;
    alldata(s).b_pt = result.b;
    alldata(s).data(:,model1_risktaking_idx) = result.probchoice; % probchoice from pt
    prev_choice = alldata(s).data(1:end-1,7);
%     prev = alldata(s).data(1:end-1,model1_risktaking_idx);
    curr = alldata(s).data(2:end,model1_risktaking_idx);
    curr_probstay = alldata(s).data(2:end,model1_risktaking_idx);
    curr_probstay(prev_choice==0) = 1-curr(prev_choice == 0);
    alldata(s).data(:,pt_probstay_idx) = [NaN; curr_probstay]; % probstay from pt
    pseudor2_study12(s,1) = result.pseudoR2;
    
    
    result = fitmodel_omnibus_persevrb(alldata(s).data,parameters_to_split);
    alldata(s).result_omnibus_apav_model = result;
    alldata(s).data(:,fullmodel_risktaking_idx) = result.probchoice; % saving probchoice in 22nd data column
    curr_model = alldata(s).data(2:end,fullmodel_risktaking_idx);
    curr_probstay_model = alldata(s).data(2:end,fullmodel_risktaking_idx);
    curr_probstay_model(prev_choice==0) = 1-curr_model(prev_choice==0);
    alldata(s).data(:,fullmodel_stay_idx) = [NaN; curr_probstay_model];

    apav_params(s,:) = result.b;
    apav_params_betalabels = alldata(1).result_omnibus_apav_model.betalabel; 
end

for s=1:length(study1)
    fprintf(sprintf('fitting model to participant %.0f of %.0f, study 1...\n',s,length(study1)))
%     result = fitmodel_omnibus_rb(study1(s).data,parameters_to_split); % rb only model for 2D
    result = fitmodel_omnibus_persevrb(study1(s).data,parameters_to_split);
    study1(s).result_omnibus_apav_model = result;
    apav_params_study1(s,:) = result.b;
    apav_params_betalabels_study1 = study1(1).result_omnibus_apav_model.betalabel;
    pseudor2_study1(s,1) = result.pseudoR2;
end

for s=1:length(study2)
    fprintf(sprintf('fitting model to participant %.0f of %.0f, study 2...\n',s,length(study2)))
%     result = fitmodel_omnibus_rb(study2(s).data,parameters_to_split); % rb only model for 2D
    result = fitmodel_omnibus_persevrb(study2(s).data,parameters_to_split);
    study2(s).result_omnibus_apav_model = result;
    apav_params_study2(s,:) = result.b;
    apav_params_betalabels_study2 = study2(1).result_omnibus_apav_model.betalabel; 
    pseudor2_study2(s,1) = result.pseudoR2;
end


%% Figure 5C & 6C: Both Persev & Bias Effects on the same plot.  
%{
predict_reward_responses = cell(length(alldata),1);
carries_info_responses = cell(length(alldata),1);

for s = 1:length(alldata)
    predict_reward_responses{s,1} = alldata(s).predictedreward;
    carries_info_responses{s,1} = alldata(s).carries_info;

end
[resp_key_ab,~,responses_carryinfo] = unique(carries_info_responses); % 1-IDK, 2-No, 3-Yes
[resp_key_pr,~,responses_pr] = unique(predict_reward_responses); % 1-IDK, 2-No, 3-Yes

response_question = responses_pr; % responses_pr or responses_carryinfo
question_text = 'Predicts Reward';

splitpersev = apav_params(response_question==1,strcmp(apav_params_betalabels,'{\Delta} persev'));
splitrb = apav_params(response_question==1,strcmp(apav_params_betalabels,'{\Delta} risky bias'));
%}

% UNCOMMENT FOR ORIGINAL
% 
splitpersev = apav_params(:,strcmp(apav_params_betalabels,'{\Delta} persev'));
splitrb = apav_params(:,strcmp(apav_params_betalabels,'{\Delta} risky bias'));

% correlate the two effects
[rho_persevrb,p_persevrb] = corr(splitpersev,splitrb,'Type','Spearman');
fprintf(sprintf('Spearman rho = %.03f, p = %.03f',rho_persevrb,p_persevrb))

% Making a pretty bar plot for parameter diffs
orange_color_rb = [219 139 46]/255;
periwinkle_color_persev = [110 133 195]/255;

ylabel_text = '\delta difference parameter estimate'; % for experiments 3 and 4
% ylabel_text = '\delta difference parameter [B ending - A ending]'; % for experiments 5 and 6
xaxis_labels = {'Risky Bias','Perseveration'};
yaxislimits = [-1.5 1.5];
bar_colors_vec1vec2 = [orange_color_rb;periwinkle_color_persev];
vec1vec2_params = {'delta_riskybias','delta_persev'}


modelbased_barplot(splitrb,splitpersev,ylabel_text,xaxis_labels,yaxislimits,bar_colors_vec1vec2,vec1vec2_params);


riskybias_diff = splitrb;
persev_diff = splitpersev;

% source_data_5c1 = table(riskybias_diff,persev_diff);
% writetable(source_data_5c1,'source_data_5c1.csv')

% source_data_5c2 = table(riskybias_diff,persev_diff);
% writetable(source_data_5c2,'source_data_5c2.csv')

% source_data_6c1 = table(riskybias_diff,persev_diff);
% writetable(source_data_6c1,'source_data_6c1.csv')

% source_data_6c2 = table(riskybias_diff,persev_diff);
% writetable(source_data_6c2,'source_data_6c2.csv')

% source_data_7b1 = table(riskybias_diff,persev_diff);
% writetable(source_data_7b1,'source_data_7b1.csv')

% source_data_7b2 = table(riskybias_diff,persev_diff);
% writetable(source_data_7b2,'source_data_7b2.csv')

%% Figure 2D: Risk Taking Main effect (model-based): Studies 1 and 2 separately
% remember that for Figure 2D in the main paper, you're relying on the fit from just PT + rb
% + dRb, not the full model


rb_study1 = apav_params_study1(:,strcmp(apav_params_betalabels_study1,'risky bias'));
rb_study2 = apav_params_study2(:,strcmp(apav_params_betalabels_study2,'risky bias'));

param_idx = rb_study1;
mean_vec1 = mean(param_idx,'omitnan');
params_se = (std(param_idx)./sqrt(sum(~isnan(param_idx))))';
fprintf(sprintf('Risky bias parameter %s: (%.03f %s %.03f)\n',...
    study_title_1,mean_vec1,char(177),params_se(1)))
param_idx = rb_study2;
mean_vec1 = mean(param_idx,'omitnan');
params_se = (std(param_idx)./sqrt(sum(~isnan(param_idx))))';
fprintf(sprintf('Risky bias parameter %s: (%.03f %s %.03f)\n',...
    study_title_2,mean_vec1,char(177),params_se(1)))

splitrb_study1 = apav_params_study1(:,strcmp(apav_params_betalabels_study1,'{\Delta} risky bias'));
splitrb_study2 = apav_params_study2(:,strcmp(apav_params_betalabels_study2,'{\Delta} risky bias'));

ylabel_text = '\delta riskybias difference parameter estimate'; % for experiments 3 and 4
% ylabel_text = '\delta difference parameter [B ending - A ending]'; % for experiments 5 and 6
xaxis_labels = {study_title_1,study_title_2};
yaxislimits = [-1.5 1.5];
bar_colors_vec1vec2 = [orange_color_rb;orange_color_rb];
vec1vec2_params = {'delta_riskybias_exp1','delta_riskybias_exp2'};

modelbased_barplot(splitrb_study1,splitrb_study2,ylabel_text,xaxis_labels,yaxislimits,bar_colors_vec1vec2,vec1vec2_params);

% save the source data

riskybias_diff_study1 = splitrb_study1;
riskybias_diff_study2 = splitrb_study2;

source_data_2d = table(riskybias_diff_study1,riskybias_diff_study2);
% writetable(source_data_2d,'source_data_2d.csv')


[p_exp3to1,~] = ranksum(splitrb_study1,splitrb)
[p_exp3to2,~] = ranksum(splitrb_study2,splitrb)
% [p_exp34to12,~] = ranksum(splitrb_study1,splitrb)



%% Figure 3D: Perseveration Main effect (model-based): Studies 1 and 2 separately
splitpersev_study1 = apav_params_study1(:,strcmp(apav_params_betalabels_study1,'{\Delta} persev'));
splitpersev_study2 = apav_params_study2(:,strcmp(apav_params_betalabels_study2,'{\Delta} persev'));

ylabel_text = '\delta perseveration difference parameter estimate'; % for experiments 3 and 4
% ylabel_text = '\delta difference parameter [B ending - A ending]'; % for experiments 5 and 6
xaxis_labels = {study_title_1,study_title_2};
yaxislimits = [-1.5 1.5];
bar_colors_vec1vec2 = [periwinkle_color_persev;periwinkle_color_persev];
vec1vec2_params = {'delta_persev_exp1','delta_persev_exp2'};


modelbased_barplot(splitpersev_study1,splitpersev_study2,ylabel_text,xaxis_labels,yaxislimits,bar_colors_vec1vec2,vec1vec2_params);

perseveration_diff_study1 = splitpersev_study1;
perseveration_diff_study2 = splitpersev_study2;

source_data_3d = table(perseveration_diff_study1,perseveration_diff_study2);
% writetable(source_data_3d,'source_data_3d.csv')

%% Figure 2C: Gambling curve (Rare vs. Common trials)

nbins = 2;
tgambin_common=NaN(length(alldata),nbins);
tgambin_rare=NaN(length(alldata),nbins);
med_gam = NaN(length(alldata),1);

for s=1:length(alldata)
    t=alldata(s).data;  
    common = t(:,16)==1; rare = t(:,16)~=1;
    tgam_common = t(common,:); tgam_rare = t(rare,:);

    med_gam(s,1) = 0;

    % if 50% is lower than 1st quartile, then exclude
    quartiles_stay = quantile(t(:,20),3);
    if quartiles_stay(1) >= 0.5
        continue
    end
    if quartiles_stay(3) <= 0.5
        continue
    end

    percentile_common = round(tgam_common(:,20)*(nbins-1)-med_gam(s,1))+1;
    percentile_rare = round(tgam_rare(:,20)*(nbins-1)-med_gam(s,1))+1;
        
    for n = 1:nbins
        
        tgambin_common(s,n) = mean(tgam_common(percentile_common==n,7),'omitnan').*100;
        tgambin_rare(s,n) = mean(tgam_rare(percentile_rare==n,7),'omitnan').*100;
 
    end
end

ylabel_text = 'Chose risky option (%)';
risky_or_stay = 'risky';

choicecurve_halves(tgambin_common,tgambin_rare,ylabel_text,risky_or_stay,nbins);

% save the source data
common_lowp = tgambin_common(:,1);
common_highp = tgambin_common(:,2);
rare_lowp = tgambin_rare(:,1);
rare_highp = tgambin_rare(:,2);

source_data_2c = table(common_lowp,common_highp,rare_lowp,rare_highp);
% writetable(source_data_2c,'source_data_2c.csv')

%% Figure 3C: Stay curve (Rare vs. Common trials)

nbins = 2;
tstaybin_common=NaN(length(alldata),nbins);
tstaybin_rare=NaN(length(alldata),nbins);
med_stay = NaN(length(alldata),1);

for s=1:length(alldata)
    t=alldata(s).data;
    
    t(:,24) = [NaN ; diff(t(:,7))==0]; % stayed = 1; switched = 0;
    common = t(:,16)==1; rare = t(:,16)~=1;
    tstay_common = t(common,:); tstay_rare = t(rare,:);
    
    med_stay(s,1) = 0;

    % if 50% is lower than 1st quartile, then exclude
    quartiles_stay = quantile(t(:,pt_probstay_idx),3);
    if quartiles_stay(1) >= 0.5
        continue
    end
    if quartiles_stay(3) <= 0.5
        continue
    end

    percentile_common = round(tstay_common(:,pt_probstay_idx)*(nbins-1)-med_stay(s,1))+1;
    percentile_rare = round(tstay_rare(:,pt_probstay_idx)*(nbins-1)-med_stay(s,1))+1;

    for n = 1:nbins        
        tstaybin_common(s,n) = mean(tstay_common(percentile_common==n,24),'omitnan').*100;
        tstaybin_rare(s,n) = mean(tstay_rare(percentile_rare==n,24),'omitnan').*100;

    end
end

ylabel_text = 'Stayed with previous choice (%)';
risky_or_stay = 'stay';

choicecurve_halves(tstaybin_common,tstaybin_rare,ylabel_text,risky_or_stay,nbins);

% save the source data
common_lowp = tstaybin_common(:,1);
common_highp = tstaybin_common(:,2);
rare_lowp = tstaybin_rare(:,1);
rare_highp = tstaybin_rare(:,2);

source_data_3c = table(common_lowp,common_highp,rare_lowp,rare_highp);
% writetable(source_data_3c,'source_data_3c.csv')

%}
%% 8 parameter risky bias and perseveration model

pseudor2_model_1 = nan(length(alldata),1);
pseudor2_model_2 = nan(length(alldata),1);
AIC_model_1 = nan(length(alldata),1);
AIC_model_2 = nan(length(alldata),1);
apav_params_model_1 = nan(length(alldata),6);
delta_mu_model_params = nan(length(alldata),7);


model_name_1 = 'Lapse model + \delta_{lapse}';
% mu, lambda, alphagain, alphaloss, lapse comm, lapse odd

for s=1:length(alldata)
    fprintf(sprintf('fitting lapse model to participant %.0f of %.0f...\n',s,length(alldata)))
    result = fitmodel_dLapsemodel(alldata(s).data,{'lapse'}); %fits dLapse model       
    alldata(s).result_dLapsemodel = result;
    alldata(s).data(:,model1_risktaking_idx) = result.probchoice; % saving probchoice in 20th data column
    apav_params_model_1(s,:) = result.b;

    prev_choice = alldata(s).data(1:end-1,7);
    curr_probstay_model = alldata(s).data(2:end,model1_risktaking_idx);
    curr_probstay_model(prev_choice==0) = 1-curr_probstay_model(prev_choice==0);
    alldata(s).data(:,model1_stay_idx) = [NaN; curr_probstay_model];    
        
end


% mu, delta mu, lambda, alphagain, alphaloss, persev, riskybias, 

model_name_2 = 'Risky Bias Perseveration model + \delta_{\mu}';

for s=1:length(alldata)
    fprintf(sprintf('fitting delta mu model to participant %.0f of %.0f...\n',s,length(alldata)))
    result = fitmodel_omnibus_persevrb(alldata(s).data,{'\mu'});
    alldata(s).result_omnibus_apav_model_dMu = result;
    delta_mu_model_params(s,:) = result.b;
    
    alldata(s).data(:,model2_risktaking_idx) = result.probchoice; % saving probchoice in 21st data column
    prev_choice = alldata(s).data(1:end-1,7);
    
    curr_probstay_model = alldata(s).data(2:end,model2_risktaking_idx);
    curr_probstay_model(prev_choice==0) = 1-curr_probstay_model(prev_choice==0);
    alldata(s).data(:,model2_stay_idx) = [NaN; curr_probstay_model];    
        
end


%% Model predictions: Comparing different models
gambling_frequency_modelgenerated_comm = nan(length(alldata),3);
gambling_frequency_modelgenerated_odd = nan(length(alldata),3);
gambling_frequency_modelgenerated_2_comm = nan(length(alldata),3);
gambling_frequency_modelgenerated_2_odd = nan(length(alldata),3);
gambling_frequency_modelgenerated_3_comm = nan(length(alldata),3);
gambling_frequency_modelgenerated_3_odd = nan(length(alldata),3);
gambling_frequency_comm = nan(length(alldata),3);
gambling_frequency_rare = nan(length(alldata),3);
avg_gambling_change_model_1 = nan(length(alldata),2);
avg_gambling_change_realdata = nan(length(alldata),2);

for s = 1:length(alldata)
    t = alldata(s).data;
    % isolating trials where gamble is chosen
    gain_d = t(:,3)>0; mixed_d = t(:,3)==0; loss_d = t(:,3)<0;
    common = t(:,16)==1; oddball = t(:,16)~=1;
    gambling_frequency_modelgenerated_comm(s,1) = mean(t(common & gain_d,model1_risktaking_idx),'omitnan'); % common gain domain
    gambling_frequency_modelgenerated_odd(s,1) = mean(t(oddball & gain_d,model1_risktaking_idx),'omitnan'); % oddball gain   
    gambling_frequency_modelgenerated_comm(s,2) = mean(t(common & mixed_d,model1_risktaking_idx),'omitnan'); % common mixed domain
    gambling_frequency_modelgenerated_odd(s,2) = mean(t(oddball & mixed_d,model1_risktaking_idx),'omitnan'); % oddball mixed
    gambling_frequency_modelgenerated_comm(s,3) = mean(t(common & loss_d,model1_risktaking_idx),'omitnan'); % common loss domain
    gambling_frequency_modelgenerated_odd(s,3) = mean(t(oddball & loss_d,model1_risktaking_idx),'omitnan'); % oddball loss
    avg_gambling_change_model_1(s,1) = mean(t(common,20),'omitnan');
    avg_gambling_change_model_1(s,2) = mean(t(oddball,20),'omitnan');
    
    gambling_frequency_modelgenerated_2_comm(s,1) = mean(t(common & gain_d,model2_risktaking_idx),'omitnan'); % common gain domain
    gambling_frequency_modelgenerated_2_odd(s,1) = mean(t(oddball & gain_d,model2_risktaking_idx),'omitnan'); % oddball gain   
    gambling_frequency_modelgenerated_2_comm(s,2) = mean(t(common & mixed_d,model2_risktaking_idx),'omitnan'); % common mixed domain
    gambling_frequency_modelgenerated_2_odd(s,2) = mean(t(oddball & mixed_d,model2_risktaking_idx),'omitnan'); % oddball mixed
    gambling_frequency_modelgenerated_2_comm(s,3) = mean(t(common & loss_d,model2_risktaking_idx),'omitnan'); % common loss domain
    gambling_frequency_modelgenerated_2_odd(s,3) = mean(t(oddball & loss_d,model2_risktaking_idx),'omitnan'); % oddball loss    

    gambling_frequency_modelgenerated_3_comm(s,1) = mean(t(common & gain_d,fullmodel_risktaking_idx),'omitnan'); % common gain domain
    gambling_frequency_modelgenerated_3_odd(s,1) = mean(t(oddball & gain_d,fullmodel_risktaking_idx),'omitnan'); % oddball gain   
    gambling_frequency_modelgenerated_3_comm(s,2) = mean(t(common & mixed_d,fullmodel_risktaking_idx),'omitnan'); % common mixed domain
    gambling_frequency_modelgenerated_3_odd(s,2) = mean(t(oddball & mixed_d,fullmodel_risktaking_idx),'omitnan'); % oddball mixed
    gambling_frequency_modelgenerated_3_comm(s,3) = mean(t(common & loss_d,fullmodel_risktaking_idx),'omitnan'); % common loss domain
    gambling_frequency_modelgenerated_3_odd(s,3) = mean(t(oddball & loss_d,fullmodel_risktaking_idx),'omitnan'); % oddball loss      
    
    gambling_frequency_comm(s,1) = mean(t(common & gain_d,7),'omitnan'); % common gain domain
    gambling_frequency_rare(s,1) = mean(t(oddball & gain_d,7),'omitnan'); % oddball gain   
    gambling_frequency_comm(s,2) = mean(t(common & mixed_d,7),'omitnan'); % common mixed domain
    gambling_frequency_rare(s,2) = mean(t(oddball & mixed_d,7),'omitnan'); % oddball mixed
    gambling_frequency_comm(s,3) = mean(t(common & loss_d,7),'omitnan'); % common loss domain
    gambling_frequency_rare(s,3) = mean(t(oddball & loss_d,7),'omitnan'); % oddball loss
    avg_gambling_change_realdata(s,1) = mean(t(common,7),'omitnan');
    avg_gambling_change_realdata(s,2) = mean(t(oddball,7),'omitnan');
end

gambling_frequency_comm = gambling_frequency_comm * 100;
gambling_frequency_rare = gambling_frequency_rare * 100;
gambling_frequency_modelgenerated_comm = gambling_frequency_modelgenerated_comm * 100;
gambling_frequency_modelgenerated_odd = gambling_frequency_modelgenerated_odd * 100;
gambling_frequency_modelgenerated_2_comm = gambling_frequency_modelgenerated_2_comm * 100;
gambling_frequency_modelgenerated_2_odd = gambling_frequency_modelgenerated_2_odd * 100;
gambling_frequency_modelgenerated_3_comm = gambling_frequency_modelgenerated_3_comm * 100;
gambling_frequency_modelgenerated_3_odd = gambling_frequency_modelgenerated_3_odd * 100;


% save the source data
gain_rarecommdiff_realdata = gambling_frequency_rare(:,1)-gambling_frequency_comm(:,1);
mixed_rarecommdiff_realdata = gambling_frequency_rare(:,2)-gambling_frequency_comm(:,2);
loss_rarecommdiff_realdata = gambling_frequency_rare(:,3)-gambling_frequency_comm(:,3);

gain_rarecommdiff_rbpersevdiff = gambling_frequency_modelgenerated_3_odd(:,1)-gambling_frequency_modelgenerated_3_comm(:,1);
mixed_rarecommdiff_rbpersevdiff = gambling_frequency_modelgenerated_3_odd(:,2)-gambling_frequency_modelgenerated_3_comm(:,2);
loss_rarecommondiff_rbpersevdiff = gambling_frequency_modelgenerated_3_odd(:,3)-gambling_frequency_modelgenerated_3_comm(:,3);

gain_rarecommdiff_lapsemodel = gambling_frequency_modelgenerated_odd(:,1)-gambling_frequency_modelgenerated_comm(:,1);
mixed_rarecommdiff_lapsemodel = gambling_frequency_modelgenerated_odd(:,2)-gambling_frequency_modelgenerated_comm(:,2);
loss_rarecommdiff_lapsemodel = gambling_frequency_modelgenerated_odd(:,3)-gambling_frequency_modelgenerated_comm(:,3);

gain_rarecommdiff_stochasticitymodel = gambling_frequency_modelgenerated_2_odd(:,1)-gambling_frequency_modelgenerated_2_comm(:,1);
mixed_rarecommdiff_stochasticitymodel = gambling_frequency_modelgenerated_2_odd(:,2)-gambling_frequency_modelgenerated_2_comm(:,2);
loss_rarecommdiff_stochasticitymodel = gambling_frequency_modelgenerated_2_odd(:,3)-gambling_frequency_modelgenerated_2_comm(:,3);


source_data_4a = table(gain_rarecommdiff_realdata,mixed_rarecommdiff_realdata,loss_rarecommdiff_realdata,...
    gain_rarecommdiff_rbpersevdiff,mixed_rarecommdiff_rbpersevdiff,loss_rarecommondiff_rbpersevdiff,...
    gain_rarecommdiff_lapsemodel,mixed_rarecommdiff_lapsemodel,loss_rarecommdiff_lapsemodel,...
    gain_rarecommdiff_stochasticitymodel,mixed_rarecommdiff_stochasticitymodel,loss_rarecommdiff_stochasticitymodel);

% writetable(source_data_4a,'source_data_4a.csv')

source_data_5b1 = table(gain_rarecommdiff_realdata,mixed_rarecommdiff_realdata,loss_rarecommdiff_realdata,...
    gain_rarecommdiff_rbpersevdiff,mixed_rarecommdiff_rbpersevdiff,loss_rarecommondiff_rbpersevdiff);

% writetable(source_data_5b1,'source_data_5b1.csv')

source_data_6b1 = table(gain_rarecommdiff_realdata,mixed_rarecommdiff_realdata,loss_rarecommdiff_realdata,...
    gain_rarecommdiff_rbpersevdiff,mixed_rarecommdiff_rbpersevdiff,loss_rarecommondiff_rbpersevdiff);

% writetable(source_data_6b1,'source_data_6b1.csv')

source_data_7a1 = table(gain_rarecommdiff_realdata,mixed_rarecommdiff_realdata,loss_rarecommdiff_realdata,...
    gain_rarecommdiff_rbpersevdiff,mixed_rarecommdiff_rbpersevdiff,loss_rarecommondiff_rbpersevdiff);

% writetable(source_data_7a1,'source_data_7a1.csv')


%% model predictions for stay effect


stay_frequency_modelgenerated_comm = zeros(length(alldata),3);
stay_frequency_modelgenerated_rare = zeros(length(alldata),3);
stay_frequency_modelgenerated_2_comm = zeros(length(alldata),3);
stay_frequency_modelgenerated_2_rare = zeros(length(alldata),3);
stay_frequency_modelgenerated_3_comm = zeros(length(alldata),3);
stay_frequency_modelgenerated_3_rare = zeros(length(alldata),3);
stay_frequency_realdata_comm = zeros(length(alldata),3);
stay_frequency_realdata_rare = zeros(length(alldata),3);

for s = 1:length(alldata)
    t = alldata(s).data;
    % isolating trials where stay happens
    common = t(:,16)==1; oddball = t(:,16)~=1; 
    gain_d = t(:,3)>0; mixed_d = t(:,3)==0; loss_d = t(:,3)<0;
    
    stay_frequency_modelgenerated_comm(s,1) = mean(t(common & gain_d,model1_stay_idx),'omitnan'); % common gain domain
    stay_frequency_modelgenerated_rare(s,1) = mean(t(oddball & gain_d,model1_stay_idx),'omitnan'); % oddball gain   
    stay_frequency_modelgenerated_comm(s,2) = mean(t(common & mixed_d,model1_stay_idx),'omitnan'); % common mixed domain
    stay_frequency_modelgenerated_rare(s,2) = mean(t(oddball & mixed_d,model1_stay_idx),'omitnan'); % oddball mixed
    stay_frequency_modelgenerated_comm(s,3) = mean(t(common & loss_d,model1_stay_idx),'omitnan'); % common loss domain
    stay_frequency_modelgenerated_rare(s,3) = mean(t(oddball & loss_d,model1_stay_idx),'omitnan'); % oddball loss

    stay_frequency_modelgenerated_2_comm(s,1) = mean(t(common & gain_d,model2_stay_idx),'omitnan'); % common gain domain
    stay_frequency_modelgenerated_2_rare(s,1) = mean(t(oddball & gain_d,model2_stay_idx),'omitna'); % oddball gain   
    stay_frequency_modelgenerated_2_comm(s,2) = mean(t(common & mixed_d,model2_stay_idx),'omitnan'); % common mixed domain
    stay_frequency_modelgenerated_2_rare(s,2) = mean(t(oddball & mixed_d,model2_stay_idx),'omitnan'); % oddball mixed
    stay_frequency_modelgenerated_2_comm(s,3) = mean(t(common & loss_d,model2_stay_idx),'omitnan'); % common loss domain
    stay_frequency_modelgenerated_2_rare(s,3) = mean(t(oddball & loss_d,model2_stay_idx),'omitnan'); % oddball loss
        
    stay_frequency_modelgenerated_3_comm(s,1) = mean(t(common & gain_d,fullmodel_stay_idx),'omitnan'); % common gain domain
    stay_frequency_modelgenerated_3_rare(s,1) = mean(t(oddball & gain_d,fullmodel_stay_idx),'omitna'); % oddball gain   
    stay_frequency_modelgenerated_3_comm(s,2) = mean(t(common & mixed_d,fullmodel_stay_idx),'omitnan'); % common mixed domain
    stay_frequency_modelgenerated_3_rare(s,2) = mean(t(oddball & mixed_d,fullmodel_stay_idx),'omitnan'); % oddball mixed
    stay_frequency_modelgenerated_3_comm(s,3) = mean(t(common & loss_d,fullmodel_stay_idx),'omitnan'); % common loss domain
    stay_frequency_modelgenerated_3_rare(s,3) = mean(t(oddball & loss_d,fullmodel_stay_idx),'omitnan'); % oddball loss
    
    stayed = diff(t(:,7))==0; %repeated choice to stimulus
    gain_d = t(2:end,3)>0; mixed_d = t(2:end,3)==0; loss_d = t(2:end,3)<0;
    common = t(2:end,16)==1; oddball = t(2:end,16)~=1;
       
    stay_frequency_realdata_comm(s,1) = mean(stayed(common & gain_d),'omitnan'); % common gain domain
    stay_frequency_realdata_rare(s,1) = mean(stayed(oddball & gain_d),'omitnan'); % oddball gain   
    stay_frequency_realdata_comm(s,2) = mean(stayed(common & mixed_d),'omitnan'); % common mixed domain
    stay_frequency_realdata_rare(s,2) = mean(stayed(oddball & mixed_d),'omitnan'); % oddball mixed
    stay_frequency_realdata_comm(s,3) = mean(stayed(common & loss_d),'omitnan'); % common loss domain
    stay_frequency_realdata_rare(s,3) = mean(stayed(oddball & loss_d),'omitnan'); % oddball loss

end

stay_frequency_realdata_comm = stay_frequency_realdata_comm * 100;
stay_frequency_realdata_rare = stay_frequency_realdata_rare * 100;
stay_frequency_modelgenerated_comm = stay_frequency_modelgenerated_comm * 100;
stay_frequency_modelgenerated_rare = stay_frequency_modelgenerated_rare * 100;
stay_frequency_modelgenerated_2_comm = stay_frequency_modelgenerated_2_comm * 100;
stay_frequency_modelgenerated_2_rare = stay_frequency_modelgenerated_2_rare * 100;
stay_frequency_modelgenerated_3_comm = stay_frequency_modelgenerated_3_comm * 100;
stay_frequency_modelgenerated_3_rare = stay_frequency_modelgenerated_3_rare * 100;

% save the source data
stay_gain_rarecommdiff_realdata = stay_frequency_realdata_rare(:,1)-stay_frequency_realdata_comm(:,1);
stay_mixed_rarecommdiff_realdata = stay_frequency_realdata_rare(:,2)-stay_frequency_realdata_comm(:,2);
stay_loss_rarecommdiff_realdata = stay_frequency_realdata_rare(:,3)-stay_frequency_realdata_comm(:,3);

stay_gain_rarecommdiff_rbpersevdiff = stay_frequency_modelgenerated_3_rare(:,1)-stay_frequency_modelgenerated_3_comm(:,1);
stay_mixed_rarecommdiff_rbpersevdiff = stay_frequency_modelgenerated_3_rare(:,2)-stay_frequency_modelgenerated_3_comm(:,2);
stay_loss_rarecommondiff_rbpersevdiff = stay_frequency_modelgenerated_3_rare(:,3)-stay_frequency_modelgenerated_3_comm(:,3);

stay_gain_rarecommdiff_lapsemodel = stay_frequency_modelgenerated_rare(:,1)-stay_frequency_modelgenerated_comm(:,1);
stay_mixed_rarecommdiff_lapsemodel = stay_frequency_modelgenerated_rare(:,2)-stay_frequency_modelgenerated_comm(:,2);
stay_loss_rarecommdiff_lapsemodel = stay_frequency_modelgenerated_rare(:,3)-stay_frequency_modelgenerated_comm(:,3);

stay_gain_rarecommdiff_stochasticitymodel = stay_frequency_modelgenerated_2_rare(:,1)-stay_frequency_modelgenerated_2_comm(:,1);
stay_mixed_rarecommdiff_stochasticitymodel = stay_frequency_modelgenerated_2_rare(:,2)-stay_frequency_modelgenerated_2_comm(:,2);
stay_loss_rarecommdiff_stochasticitymodel = stay_frequency_modelgenerated_2_rare(:,3)-stay_frequency_modelgenerated_2_comm(:,3);


source_data_4b = table(stay_gain_rarecommdiff_realdata,stay_mixed_rarecommdiff_realdata,stay_loss_rarecommdiff_realdata,...
    stay_gain_rarecommdiff_rbpersevdiff,stay_mixed_rarecommdiff_rbpersevdiff,stay_loss_rarecommondiff_rbpersevdiff,...
    stay_gain_rarecommdiff_lapsemodel,stay_mixed_rarecommdiff_lapsemodel,stay_loss_rarecommdiff_lapsemodel,...
    stay_gain_rarecommdiff_stochasticitymodel,stay_mixed_rarecommdiff_stochasticitymodel,stay_loss_rarecommdiff_stochasticitymodel);
% writetable(source_data_4b,'source_data_4b.csv')

source_data_5b2 = table(stay_gain_rarecommdiff_realdata,stay_mixed_rarecommdiff_realdata,stay_loss_rarecommdiff_realdata,...
    stay_gain_rarecommdiff_rbpersevdiff,stay_mixed_rarecommdiff_rbpersevdiff,stay_loss_rarecommondiff_rbpersevdiff);
% writetable(source_data_5b2,'source_data_5b2.csv')

source_data_6b2 = table(stay_gain_rarecommdiff_realdata,stay_mixed_rarecommdiff_realdata,stay_loss_rarecommdiff_realdata,...
    stay_gain_rarecommdiff_rbpersevdiff,stay_mixed_rarecommdiff_rbpersevdiff,stay_loss_rarecommondiff_rbpersevdiff);
% writetable(source_data_6b2,'source_data_6b2.csv')

source_data_7a2 = table(stay_gain_rarecommdiff_realdata,stay_mixed_rarecommdiff_realdata,stay_loss_rarecommdiff_realdata,...
    stay_gain_rarecommdiff_rbpersevdiff,stay_mixed_rarecommdiff_rbpersevdiff,stay_loss_rarecommondiff_rbpersevdiff);
% writetable(source_data_7a2,'source_data_7a2.csv')

%% Figure 4, 5B or 6B: Lapse model and delta mu model risk taking predictions comparison 
% Comparing Model Generated and Real data Rare - Common Gambling Rates
% Main Effects

rarecommdiff_risktaking_realdata = gambling_frequency_rare - gambling_frequency_comm;
competing_modelpred{1} = gambling_frequency_modelgenerated_3_odd-gambling_frequency_modelgenerated_3_comm;
competing_modelpred{2} = gambling_frequency_modelgenerated_2_odd-gambling_frequency_modelgenerated_2_comm;
competing_modelpred{3}  = gambling_frequency_modelgenerated_odd-gambling_frequency_modelgenerated_comm;


competing_model_name = {sprintf('%s: (n = %.0f)',study_title,length(alldata)),model_name,model_name_2,model_name_1};

ylabel_name_risk = {'Rare - Common trials:';'Chose risky option (%)'};
% ylabel_name_risk = {'B ending - A ending trials:';'Chose risky option (%)'};

yaxislimits = [-12 12];

colorscheme = [0.47 0.25 0.80;0.83 0.14 0.14;1.00 0.54 0.00]; % color map for model predictions
barcolor = '#DED0BF';

% num_modelpredictions = 1; % run if you only want to show the full model predictions on the plot
num_modelpredictions = length(competing_modelpred); % run if you want to show model predictions from lapse model and split mu model too 


threebarplot_modelpredictions_ontop(rarecommdiff_risktaking_realdata,ylabel_name_risk,yaxislimits,competing_modelpred,competing_model_name,colorscheme,num_modelpredictions,barcolor,study_title);

% creating the same bar plot for perseveration effect


rarecommdiff_stay_realdata = stay_frequency_realdata_rare - stay_frequency_realdata_comm;
competing_modelpred_stay{1} = stay_frequency_modelgenerated_3_rare-stay_frequency_modelgenerated_3_comm;
competing_modelpred_stay{2} = stay_frequency_modelgenerated_2_rare-stay_frequency_modelgenerated_2_comm;
competing_modelpred_stay{3}  = stay_frequency_modelgenerated_rare-stay_frequency_modelgenerated_comm;

ylabel_name_stay = {'Rare - Common trials:'; 'Stayed with previous option (%)'};
% ylabel_name_stay = {'B ending - A ending trials:'; 'Stayed with previous option (%)'}; % For experiments 5 and 6


barcolor = '#D3D5DD';


threebarplot_modelpredictions_ontop(rarecommdiff_stay_realdata,ylabel_name_stay,yaxislimits,competing_modelpred_stay,competing_model_name,colorscheme,num_modelpredictions,barcolor,study_title);

toc

%% PLOTTING FUNCTIONS %%%%%%

function b = sixbarplot(common_frequencies,rare_frequencies,list_yloc,ylabel_name,study_title)

    [p_OCgaingam,~] = signrank(rare_frequencies(:,1)-common_frequencies(:,1));
    [p_OCmixedgam,~] = signrank(rare_frequencies(:,2)-common_frequencies(:,2));
    [p_OClossgam,~] = signrank(rare_frequencies(:,3)-common_frequencies(:,3));
    [sig_stars,fz] = sigstar([p_OCgaingam,p_OCmixedgam,p_OClossgam]);;
    
    common_se = std(common_frequencies,'omitnan')./sqrt(sum(~isnan(common_frequencies)));
    oddballs_se = std(rare_frequencies,'omitnan')./sqrt(sum(~isnan(rare_frequencies)));
    OCdiff_se = std(rare_frequencies-common_frequencies,'omitnan')./sqrt(sum(~isnan(rare_frequencies-common_frequencies)));
    se_bars = [common_se; oddballs_se];

    grouped_bars = vertcat(nanmean(common_frequencies),nanmean(rare_frequencies))';

    trial_types = {'gain','mixed','loss'};
    for i = 1:3
        [OCdiffp_value,~] = signrank(rare_frequencies(:,i)-common_frequencies(:,i));
        fprintf(sprintf('Rare - Common rate diff %s trials %s: (%.02f %s %.02f%s, p = %.03f)\n',...
            study_title,trial_types{i},mean(rare_frequencies(:,i)-common_frequencies(:,i),'omitnan'),char(177),...
            OCdiff_se(:,i),'%%',OCdiffp_value))
    end
%     fprintf(sprintf('%s Bayes Factor 01: %.02f\n',study_title_1,bf01_gameffect_study1))

    figure1 = figure('color',[1 1 1]); 
    axes1 = axes('Parent',figure1);
    width=600; height=400;
    set(gcf,'position',[10,10,width,height])
    blue_color = [0.066666666666667 0.227450980392157 0.674509803921569];
    red_color = [0.768627450980392 0.196078431372549 0.196078431372549];
    b = bar(grouped_bars,'grouped','EdgeColor','black','LineWidth',1.5,'BarWidth', 0.8); hold on;
    set(b(1),'FaceColor',blue_color);
    set(b(2),'FaceColor',red_color); 
    set(axes1, 'box','off');
    for x = 1:3
        swarmchart(repmat(b(1).XData(x)-0.15, length(common_frequencies), 1),common_frequencies(:,x),4,...
            'MarkerFaceColor','#adbaff','MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.15);
        swarmchart(repmat(b(2).XData(x)+0.15, length(rare_frequencies), 1),rare_frequencies(:,x),4,...
            'MarkerFaceColor','#ffd4d4','MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.15);
    end
    % hline = refline(0,50); hline.Color = 'black'; hline.LineStyle = '--'; 
    [ngroups,nbars] = size(grouped_bars);
    
    ctr = nan(nbars,3); ydt = nan(nbars,3);
    for i = 1:2
        ctr(i,:) = b(i).XData + b(i).XOffset';    
        ydt(i,:) = b(i).YData;    
    end
    errorbar(ctr, ydt, se_bars, '.black','LineWidth',2,'CapSize',7); hold on; 
    % box(axes1,'on');
    set(gca,'xtick',1:3,'xticklabel',{'Gain trials','Mixed trials','Loss trials'}); hold on;
    ylim([0 100]); yticks(0:25:100);
    xlim([0.3 3.7]);
    text(0.35,93,sprintf('n=%.0f',length(common_frequencies)),'fontsize',14,'HorizontalAlignment', 'left');
    ylabel(sprintf('%s',ylabel_name)); 
%     list_yloc = [75,65,45];
    for i = 1:ngroups
        xloc = [b(1).XData(i)+b(1).XOffset,b(2).XData(i)+b(2).XOffset];
        yloc = [list_yloc(i),list_yloc(i)];
        plot(xloc,yloc,...
            'color', 'k',...
            'linewidth', 1);
        text(mean(xloc),mean(yloc)+0.8,sig_stars(i),...
            'fontsize',fz(i),...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'middle');
    end
    legend({'Common','Rare'},'NumColumns',1);
    hold off
end


function b = modelfree_avg_fourbarplot(common_frequencies,rare_frequencies,overall_study1,overall_study2,list_yloc,ylabel_name,study_title_1,study_title_2,behav_measure)
 
    common_se = std(common_frequencies,'omitnan')./sqrt(sum(~isnan(common_frequencies)));
    oddballs_se = std(rare_frequencies,'omitnan')./sqrt(sum(~isnan(rare_frequencies)));
    
    ob_diff_study1 = rare_frequencies(:,1)-common_frequencies(:,1);
    ob_diff_study2 = rare_frequencies(:,2)-common_frequencies(:,2);
    
    [p_study1,~] = signrank(ob_diff_study1);
    [p_study2,~] = signrank(ob_diff_study2);
%     [h,p_study1] = ttest(ob_diff_study1);
%     [h,p_study2] = ttest(ob_diff_study2);
    [sig_stars,fontsize] = sigstar([p_study1 p_study2]);
    [p_study1study2_gam, ~] = ranksum(overall_study1,overall_study2);
    [sig_stars_group,fontsize_group] = sigstar(p_study1study2_gam);
    
    overall_study1_se = std(overall_study1,'omitnan')./sqrt(sum(~isnan(overall_study1)));
    overall_study2_se = std(overall_study2,'omitnan')./sqrt(sum(~isnan(overall_study2)));

    
    se_bars = [common_se; oddballs_se];
    grouped_bars = vertcat(mean(common_frequencies,'omitnan'),mean(rare_frequencies,'omitnan'))';
    
    gam_obdiff_se_study1 = std(ob_diff_study1,'omitnan')./sqrt(sum(~isnan(ob_diff_study1)));
    gam_obdiff_se_study2 = std(ob_diff_study2,'omitnan')./sqrt(sum(~isnan(ob_diff_study2)));
    
%     [bf10_gameffect_study1,~] = ttest_bf(ob_diff_study1,'tail','right');
    [bf10_gameffect_study1,~] = ttest_bf(ob_diff_study1);
    
    bf01_gameffect_study1 = 1/bf10_gameffect_study1;

%     [bf10_gameffect_study2,~] = ttest_bf(ob_diff_study2,'tail','right');
    [bf10_gameffect_study2,~] = ttest_bf(ob_diff_study2);
    
    bf01_gameffect_study2 = 1/bf10_gameffect_study2;
    

    fprintf(sprintf('%s average %s rate: (%.02f %s %.02f%s)\n',...
        study_title_1,behav_measure,mean(overall_study1),char(177),overall_study1_se,'%%'))

    fprintf(sprintf('%s %s rate diff: (%.02f %s %.02f%s, p = %.03f)\n',...
        study_title_1,behav_measure,mean(ob_diff_study1),char(177),gam_obdiff_se_study1,'%%',p_study1))

    fprintf(sprintf('%s Bayes Factor 01 (for the null): %.02f\n',study_title_1,bf01_gameffect_study1))
    
    fprintf(sprintf('%s average %s rate: (%.02f %s %.02f%s)\n',...
        study_title_2,behav_measure,mean(overall_study2),char(177),overall_study2_se,'%%'))

    fprintf(sprintf('%s %s rate diff: (%.02f %s %.02f%s, p = %.03f)\n',...
        study_title_2,behav_measure,mean(ob_diff_study2),char(177),gam_obdiff_se_study2,'%%',p_study2))

    fprintf(sprintf('%s Bayes Factor 01 (for the null): %.02f\n',study_title_2,bf01_gameffect_study2))
    
    
    figure1 = figure('color',[1 1 1]);
    axes1 = axes('Parent',figure1);
    width=400; height=400;
    set(gcf,'position',[10,10,width,height]);
    b = bar(grouped_bars,'grouped','EdgeColor','black','LineWidth',1.5,'BarWidth', 0.8); hold on;
    set(axes1, 'box','off'); 
    set(b(1),...
        'FaceColor',[0.066666666666667 0.227450980392157 0.674509803921569]);
    set(b(2),...
        'FaceColor',[0.768627450980392 0.196078431372549 0.196078431372549]);
    for x = 1:2
        swarmchart(repmat(b(1).XData(x)-0.15, length(common_frequencies), 1),common_frequencies(:,x),4,...
            'MarkerFaceColor','#adbaff','MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.15);
        swarmchart(repmat(b(2).XData(x)+0.15, length(rare_frequencies), 1),rare_frequencies(:,x),4,...
            'MarkerFaceColor','#ffd4d4','MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.15); hold on;
    end
    ylim([0 100]); yticks(0:25:100); hold on;
    xlim([0.5 2.5]);
    set(gca,'xtick',1:2,'xticklabel',{'Exp. 1','Exp. 2'}); hold on;
    % hline = refline(0,50); hline.Color = 'black'; hline.LineStyle = '--';
    ylabel(ylabel_name)
    [ngroups,nbars] = size(grouped_bars);
    % Get the x coordinate of the bars
    x = nan(nbars, ngroups);
    for i = 1:nbars
        ctr_x(i,:) = bsxfun(@plus, b(i).XData, b(i).XOffset');    
        ydt_y(i,:) = b(i).YData;    
    end
    errorbar(ctr_x, ydt_y, se_bars, '.black','LineWidth',2,'CapSize',7);
    hold on;
%     list_yloc = [60,56];
    for i = 1:ngroups
        xloc = [b(1).XData(i)+b(1).XOffset,b(2).XData(i)+b(2).XOffset];
        yloc = [list_yloc(i),list_yloc(i)];
        plot(xloc,yloc,...
            'color', 'k',...
            'linewidth', 1);
        text(mean(xloc),mean(yloc)+0.8,sig_stars(i),...
            'fontsize',fontsize(i),...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'middle');
    end
    xloc_group = [b(1).XData(1)+b(1).XOffset,b(2).XData(2)+b(2).XOffset];
    yloc_group = [70 70];
        plot(xloc_group,yloc_group,...
            'color', 'k',...
            'linewidth', 1);
        text(mean(xloc_group),mean(yloc_group)+2,sig_stars_group(1),...
            'fontsize',fontsize_group(1),...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'middle');
    legend({'Common','Rare'});
    hold off
end

function b = modelbased_barplot(vec1,vec2,ylabel_name,xaxislabel,ylimits,bar_colors_vec1vec2,vec1vec2_params)
    
    parameterdiffs = [vec1 vec2];
    % calculating se for the bar graph
    params_se = (std(parameterdiffs)./sqrt(sum(~isnan(parameterdiffs))))';
    bars = mean(parameterdiffs)';
    
    mean_vec1 = mean(vec1,'omitnan');
    p_vec1 = signrank(vec1); % bias diff
    
    mean_vec2 = mean(vec2,'omitnan');
    p_vec2 = signrank(vec2); % persev diff
    
    [bf10_splitrb,~] = ttest_bf(vec1);
    bf01_splitrb = 1/bf10_splitrb;
    
    [bf10_splitpersev,~] = ttest_bf(vec2);
    bf01_splitpersev = 1/bf10_splitpersev;
    
    fprintf(sprintf('%s: (%.03f %s %.03f, p = %.03f, BF01 = %.02f)\n',...
        vec1vec2_params{1},mean_vec1,char(177),params_se(1),p_vec1,bf01_splitrb))
    fprintf(sprintf('%s: (%.03f %s %.03f, p = %.03f, BF01 = %.02f)\n',...
        vec1vec2_params{2},mean_vec2,char(177), params_se(2),p_vec2,bf01_splitpersev))
    
    [sig_stars,fontsize] = sigstar([p_vec1 p_vec2]);
    
    yaxislimits = ylimits;
    parameterdiffs_plotted = parameterdiffs; % modifying points that are out of range
    parameterdiffs_plotted(parameterdiffs_plotted>max(yaxislimits)) = max(yaxislimits);
    parameterdiffs_plotted(parameterdiffs_plotted<min(yaxislimits)) = min(yaxislimits);
    
    
    figure1 = figure('color',[1 1 1]);
    axes1 = axes('Parent',figure1);
    width=350; height=400;
    % width=300; height=400;
    set(gcf,'position',[10,10,width,height]);
    hold on
    b = bar(bars,'grouped','EdgeColor','black','LineWidth',1.5,'BarWidth', 0.8,'FaceColor','flat'); 
    swarmchart(repmat(b.XData(1), length(parameterdiffs_plotted), 1),parameterdiffs_plotted(:,1),4,...
            'MarkerFaceColor',bar_colors_vec1vec2(1,:),'MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.5)
    swarmchart(repmat(b.XData(2), length(parameterdiffs_plotted), 1),parameterdiffs_plotted(:,2),4,...
            'MarkerFaceColor',bar_colors_vec1vec2(2,:),'MarkerEdgeColor','#c9c9c9','XJitter','density','XJitterWidth',0.5)
    ylim([yaxislimits(1)-0.1 yaxislimits(2)]); xlim([0.1,2.9]);
    set(gca,'xtick',1:2,'xticklabel',xaxislabel);
    ylabel(ylabel_name)
    % title(sprintf('%s',study_title));
    errorbar(bars,params_se, '.black','LineWidth',2,'CapSize',7)
    % b = bar(bars,'grouped','EdgeColor','black','LineWidth',1.5,'BarWidth', 0.8,'FaceColor','flat'); 
    clr = bar_colors_vec1vec2;
    b.CData = clr; 
    xloc = b.XData;
    yloc = b.YData;
    for i = 1:length(bars)    
        if yloc(i) < 0
            text(xloc(i),yloc(i)-1,sig_stars(i),...
                    'fontsize',fontsize(i),...
                    'HorizontalAlignment', 'center',...
                    'VerticalAlignment', 'middle');
        elseif yloc(i) > 0
            text(xloc(i),yloc(i)+1,sig_stars(i),...
                    'fontsize',fontsize(i),...
                    'HorizontalAlignment', 'center',...
                    'VerticalAlignment', 'middle'); 
        end
    end
    text(0.2,1.35,sprintf('n=%.0f',length(vec1)),'fontsize',14,'HorizontalAlignment', 'left');
    hold off

end


function b = choicecurve_halves(common_bins,rare_bins,ylabel_name,risky_or_stay,nbins)
    
    % comparing oddball and common trials on the same softmax plot
    curve_comm = [nanmean(common_bins,1)' (nanstd(common_bins)./sqrt(sum(~isnan(common_bins))))'];
    curve_rare = [nanmean(rare_bins,1)' (nanstd(rare_bins)./sqrt(sum(~isnan(rare_bins))))'];
    
    OC_lowgroup = rare_bins(:,1) - common_bins(:,1);
    OC_highgroup = rare_bins(:,2) - common_bins(:,2);
    OC_lowgroup_se = std(OC_lowgroup,'omitnan')./sqrt(sum(~isnan(OC_lowgroup)))
    OC_highgroup_se = std(OC_highgroup,'omitnan')./sqrt(sum(~isnan(OC_highgroup)))

    [p_OClowstay,~] = signrank(OC_lowgroup);
    [p_OChighstay,~] = signrank(OC_highgroup);
    
    [sig_stars,fontsize] = sigstar([p_OClowstay,p_OChighstay]);

    fprintf(sprintf('low P(%s) Rare - Common diff: (%.02f %s %.02f%s,p = %.03f)\n',...
        risky_or_stay,mean(OC_lowgroup,'omitnan'),char(177),OC_lowgroup_se,'%%',p_OClowstay))

    fprintf(sprintf('high P(%s) Rare - Common diff: (%.02f %s %.02f%s,p = %.03f)\n',...
        risky_or_stay,mean(OC_highgroup,'omitnan'),char(177),OC_highgroup_se,'%%',p_OChighstay))

    
    figure1 = figure('color',[1 1 1]);
    axes1 = axes('Parent',figure1); 
    width=400; height=400;
    set(gcf,'position',[10,10,width,height])
    hold on;
    line_comm = plot(curve_comm(:,1),'color',[0.066666666666667 0.227450980392157 0.674509803921569]);
    line_rare = plot(curve_rare(:,1), 'color',[0.768627450980392 0.196078431372549 0.196078431372549]);
    ylim([15 85]); yticks(20:10:80); 
    xlim([0.5 nbins+0.5]); 
    xticks(1:nbins);  
    xticklabels({sprintf('low P(%s)',risky_or_stay),sprintf('high P(%s)',risky_or_stay)});
    % xticklabels({'P(stay) < 0.5','P(stay) >= 0.5'});
    hline = refline(0,50); hline.Color = 'black'; hline.LineStyle = '--';
    ylabel(ylabel_name);
    % title(sprintf('Stay rate, n= %0.f',length(alldata)));
    er = errorbar(curve_comm(:,1),curve_comm(:,2),'LineWidth',2,'CapSize',5);
    er_ob = errorbar(curve_rare(:,1),curve_rare(:,2),'LineWidth',2,'CapSize',5);
    er.Color = [0.06667 0.22745 0.67451];                            
    er.LineStyle = 'none';  
    er_ob.Color = [0.76863 0.19608 0.19608];                            
    er_ob.LineStyle = 'none';
    legend('Common','Rare');
    xloc_comm = line_comm.XData; yloc_comm = line_comm.YData;
    xloc_rare = line_rare.XData; yloc_rare = line_rare.YData;
    for i = 1:2    
        text(xloc_comm(i),max(yloc_comm(i),yloc_rare(i))+2.8,sig_stars(i),...
                    'fontsize',fontsize(i),...
                    'HorizontalAlignment', 'center',...
                    'VerticalAlignment', 'middle');
    end
end


function b = threebarplot_modelpredictions_ontop(bar_data,ylabel_name,yaxislimits,competing_modelpred,competing_model_name,colorscheme,num_modelpredictions,barcolor,study_title)
    
    
    [p_OCgaingam,~] = signrank(bar_data(:,1));
    [p_OCmixedgam,~] = signrank(bar_data(:,2));
    [p_OClossgam,~] = signrank(bar_data(:,3));
    [sig_stars,fz] = sigstar([p_OCgaingam,p_OCmixedgam,p_OClossgam]);
    se_bars = std(bar_data,'omitnan')./sqrt(sum(~isnan(bar_data)));
    


    trial_types = {'gain','mixed','loss'};
    for i = 1:3
%         [OCdiffp_value,~] = signrank(bar_data(:,i),0,'tail','right');
        [OCdiffp_value,~] = signrank(bar_data(:,i));
       
        [bf10_3bar,~] = ttest_bf(bar_data(:,i));
        bf01_3bar = 1/bf10_3bar;       
        fprintf(sprintf('Rare - Comm rate diff %s trials %s: (%.02f %s %.02f%s, p = %.03f, BF01 = %.02f)\n',...
            study_title,trial_types{i},mean(bar_data(:,i),'omitnan'),char(177),...
            se_bars(:,i),'%%',OCdiffp_value,bf01_3bar))
    end    
    
    
    figure4 = figure('color',[1 1 1]); 
    axes1 = axes('Parent',figure4);
    width=350; height=400;
    set(gcf,'position',[10,10,width,height])
    b = bar(mean(bar_data,'omitnan'),'grouped','EdgeColor','black','LineWidth',1.5,'BarWidth', 0.7); hold on;
    set(b(1),'FaceColor',barcolor);
    set(axes1, 'box','off');
    
    for p = 1:num_modelpredictions
        ypoint(1,:) = mean(competing_modelpred{p},'omitnan');  
        se_bars_model = std(competing_modelpred{p},'omitnan')./sqrt(sum(~isnan(competing_modelpred{p})));
        errorbar((1:3) + p*0.08, ypoint, se_bars_model,"o", "MarkerSize",4,'Color',colorscheme(p,:),'LineWidth',2,'CapSize',7,'LineStyle','none','MarkerFaceColor',colorscheme(p,:)); hold on; 
    
    end
    
    above_max = bar_data>max(yaxislimits);
    below_min = bar_data<min(yaxislimits);
    
    bar_data_points = bar_data; % modifying points that are out of range
    bar_data_points(above_max) = max(yaxislimits);
    bar_data_points(below_min) = min(yaxislimits);
    
    
    errorbar(1:3, mean(bar_data,'omitnan'), se_bars, '.black','LineWidth',2,'CapSize',7); hold on; 
    set(gca,'xtick',1:3,'xticklabel',{'Gain trials','Mixed trials','Loss trials'}); hold on;
    ylim(yaxislimits.*1.1); yticks(-20:4:20);
    xlim([0.3 3.7]);
    xtickangle(30)
    text(0.35,93,sprintf('n=%.0f',length(bar_data)),'fontsize',14,'HorizontalAlignment', 'left');
    % ylabel(sprintf('%s',ylabel_name)); 
    ylabel(ylabel_name); 
    
    xloc = b.XData;
    yloc = b.YData;
    for x = 1:3
        swarmchart(repmat(b(1).XData(x)-0.15, length(bar_data), 1),bar_data_points(:,x),4,...
            'MarkerFaceColor','#adbaff','MarkerEdgeColor','#8f8f8f','XJitter','density','XJitterWidth',0.15);
    end
    for i = 1:size(bar_data,2)   
        if yloc(i) < 0
            text(xloc(i),yloc(i)-3,sig_stars(i),...
                    'fontsize',fz(i),...
                    'HorizontalAlignment', 'center',...
                    'VerticalAlignment', 'middle');
        elseif yloc(i) > 0
            text(xloc(i),yloc(i)+3,sig_stars(i),...Re
                    'fontsize',fz(i),...
                    'HorizontalAlignment', 'center',...
                    'VerticalAlignment', 'middle'); 
        end
    end
    % legend(competing_model_name,'NumColumns',1);
    hold off
end
