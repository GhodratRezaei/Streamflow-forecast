%% NRM Project, PART 1: finding best model for flow forecasting
% informations available: 27 year-time series of river Ebro flow, precipitation
% and temperature in the city of Tudela, Spain.
% flow is target signal to forecast, while precipitation and temperature
% are treated as exogenous signal

%% Loading Data in the workspace

load -ascii data.txt

year_training=18;
%year_validation=9;

%Dataset consist of 3 time series (flow, precipipitation and temperature) 
%divided in 2 subset. Data from 1990 to 2007 will be used for training, while 
%data from 2008 to 2016 will be used for validation

prec_training=data(1:year_training*365,4);
flow_training=data(1:year_training*365,5);
temp_training=data(1:year_training*365,6);
prec_val=data(year_training*365+1:end,4);
flow_val=data(year_training*365+1:end,5);
temp_val=data(year_training*365+1:end,6);

%% Plots analysis (based on training data)

figure;
subplot(3,1,1);
plot(flow_training,'b');
legend('flow');

subplot(3,1,2);
plot(prec_training,'r');
legend('precipitaion');

subplot(3,1,3);
plot(temp_training,'g');
legend('temperature');

% now, tt is a repetead vector for time
tt = repmat([1:365]',year_training, 1);

% obtaining the cyclostationary mean is quiete easy. Just transform
% (reshape function) the process vector into a matrix with 365 raws. 
% mean function take the average on every raw and so, on every day.
% transposing is mandatory because mean function perform the task on
% columns and not on raws

figure;

subplot(1,3,1);
plot(tt, flow_training,'.')
hold on;
CmF=mean((reshape(flow_training, 365, year_training))');
plot(CmF, 'r', 'LineWidth', 2)
legend('flow point', 'flow cyclostationary mean');

subplot(1,3,2);
plot(tt, prec_training,'.')
hold on;
CmP=mean((reshape(prec_training, 365, year_training))');
plot(CmP, 'r', 'LineWidth', 2)
legend('precipitation point', 'precipitation cyclostationary mean');

subplot(1,3,3);
plot(tt, temp_training,'.')
hold on;
CmT=mean((reshape(temp_training, 365, year_training))');
plot(mean((reshape(temp_training, 365, year_training))'), 'r', 'LineWidth', 2)
legend('temperature point', 'temperature cyclostationary mean');

% in order to make the cyclostationary mean smoother, the moving_average
% function is used for every process time series. the higher the third
% parameter the smoother the mean.

[f, F]=moving_average(flow_training, 365, 5);

% as example here the comparison:

figure;
plot(CmF, 'b');
hold on;
plot(f, 'r');
legend('simple mean', 'moving average solution');

[p, P]=moving_average(prec_training, 365, 5);
[t, Tree]=moving_average(temp_training, 365, 5);
[f_v, F_v]=moving_average(flow_val, 365, 5);
[p_v, P_v]=moving_average(prec_val, 365, 5);
[t_v, T_v]=moving_average(temp_val, 365, 5);

% compute moving average also on the variances so that it will be possible
% to obtain a normalized time series without cyclostationary mean and with
% unitary standard deviation;

[sf, Sf]=moving_average((flow_training-F).^2,365,5);
sigmaF=sqrt(Sf);
[sp, Sp]=moving_average((prec_training-P).^2,365,5);
sigmaP=sqrt(Sp);
[st, St]=moving_average((temp_training-Tree).^2,365,5);
sigmaT=sqrt(St);
[sf_v, Sf_v]=moving_average((flow_val-F_v).^2,365,5);
sigmaF_v=sqrt(Sf_v);
[sp_v, Sp_v]=moving_average((prec_val-P_v).^2,365,5);
sigmaP_v=sqrt(Sp_v);
[st_v, St_v]=moving_average((temp_val-T_v).^2,365,5);
sigmaT_v=sqrt(St_v);

%% Detrendization
Fx = (flow_training-F)./sigmaF ;
Px = (prec_training-P)./sigmaP ;
Tx = (temp_training-Tree)./sigmaT ;
Fx_v = (flow_val-F_v)./sigmaF_v ;
Px_v = (prec_val-P_v)./sigmaP_v ;
Tx_v = (temp_val-T_v)./sigmaT_v ;

%% Correlation computation

% the results show that:
% 1)the river flow is highly autocorrelated (AR is a good option) 
% 2)precipitations is a weak information for predicting river flow
%   but can be useful
% 3)temperature has a good endless correlation (negative because increasing
%   temperature flow decreases. We suppose that since Ebro river has not glacial origins: in
%   summer evaporation has a strong impact)

% including both exogenous signal in the model is a good idea: 
% ARX(n,m1,m2)

figure;
subplot(1,3,1);
correlogram(flow_training,flow_training,20);
title('Flow Autocorrelation');

subplot(1,3,2);
correlogram(flow_training,prec_training,20);
title('Flow-Precipitation correlation');

subplot(1,3,3);
correlogram(flow_training,temp_training,20);
title('Flow-Temperature correlation');



%% Best linear proper model computation
% a function had been created to automatically perform the computation of
% ARX forecast and R2 values. A big for loop for various values of model
% order is used to compare results and obtain the best forecast
% A matrix is also built to have memory of all possible models.

maxOrder=5;

R2_best=0;
best_order= [0 0 0];
solutions= zeros(1,4);

for i=1:maxOrder
    for j=0:maxOrder
        for h=0:maxOrder
            if(i>=j-1 && i>=h-1)
             [y_temp, R2_temp] = ARX_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Px,Px_v,Tx,Tx_v,i,j,h);
             solutions=[solutions;
                        R2_temp, i, j, h,];
                if(R2_temp>R2_best) R2_best=R2_temp;
                    y_best=y_temp;
                    best_order=[i j h];
                end
            end
        end
    end
end
R2_best
%% Best linear improper model computation
% so far, only proper models have been considered, i.e. models in which
% available exogenous signals at time t+1 are those at time t, t-1, t-2...
% nowadays, rain and temperature forecasts are very accurate and so improper
% models must be analyzed: this means that at time t, precipitation and temperature 
% signals of time t+1 are available.
% By simply shifting the exogenous signals by one position, (and add 
% using the last one two time, to have vector length consistency)
% improper models can be built by using same procedure as before:

Pximp=[Px(2:end);Px(end)];
Pximp_v=[Px_v(2:end);Px_v(end)];
Tximp=[Tx(2:end);Tx(end)];
Tximp_v=[Tx_v(2:end);Tx_v(end)];

R2_best_imp=0;
best_order_imp= [0 0 0];
solutions_imp= zeros(1,4);

for i=1:maxOrder
    for j=0:maxOrder
        for h=0:maxOrder
            if(i>=j-1 && i>=h-1)           
             [y_temp, R2_temp] = ARX_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Pximp,Pximp_v,Tximp,Tximp_v,i,j,h);
             solutions_imp=[solutions_imp;
                        R2_temp, i, j, h,];
                if(R2_temp>R2_best_imp) R2_best_imp=R2_temp;
                    y_best_imp=y_temp;
                    best_order_imp=[i j h];
                end
            end
        end
    end
end
R2_best_imp
%% Linear models results
% By looking at results, it is clear that the very slight improvement in
% using improper models don't justify the fact that weather forecasts cannot
% always have a 100% reliable prediction due to some unpredictable weather
% events.


% For this reasons, for linear models, only proper ones are now considered.
% the first empty raw of the solution matrix is deleted and the raws are sorted by
% ascending values of R2 and the results of corresponding model orders are
% plotted 

solutions=solutions(2:end,:);
[~,idx] = sort(solutions(:,1));
solutions_reordered = solutions(idx,:);

figure;
subplot(4,1,1);
plot(solutions_reordered(:,1),'r');
legend('R2 Values');

subplot(4,1,2);
plot(solutions_reordered(:,2));
legend('n order');

subplot(4,1,3);
plot(solutions_reordered(:,3));
legend('m1 order');

subplot(4,1,4);
plot(solutions_reordered(:,4));
legend('m2 order');

figure;
string = sprintf('ARX(%d,%d,%d) flow forecasting',best_order(1),best_order(2),best_order(3));
plot([y_best flow_val]);
title('Best Model');
legend(string,'real flow values');

%% Non linear modelling: Artificial Neural Network

% As in linear models, both proper and improper model are built.
% Since there are 2 exogenous signal, optimal number of nodes is 4.

R2_ann_best=0;
for i = 1:10
    [y_temp, R2_temp] = ANN_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Px,Px_v,Tx,Tx_v,4);
    if (R2_temp > R2_ann_best)
        R2_ann_best = R2_temp;
        y_ann_best= y_temp;
    end
end

for i = 1:10
    [y_temp, R2_temp] = ANN_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Pximp,Pximp_v,Tximp,Tximp_v,4);
    if (R2_temp > R2_ann_best)
        R2_ann_best = R2_temp;
        y_ann_best= y_temp;
    end
end

R2_ann_best
%% Non linear modeling: Classification And Regression Trees

% computation of CART model is also performed: first the general CART model
% is computed, then the resuls of optimal CART model are given (optimal
% means optimal minimum number of elements in one leaf)

Ycart = Fx(2:end);   
Xcart = [ Fx(1:end-1), Px(1:end-1), Tx(1:end-1)];
Xcart_v = [ Fx_v(1:end-1), Px_v(1:end-1), Tx_v(1:end-1)]; 

Tree = fitrtree(Xcart,Ycart);
Ycart_v = predict(Tree,Xcart_v);

Ycart_v = [Fx_v(1); Ycart_v];
y_cart = Ycart_v.*sigmaF_v + F_v;
R2_cart = 1 - sum( (flow_val( 2 : end ) - y_cart( 2 : end )).^2 ) / sum( (flow_val( 2 : end )-F_v(2:end) ).^2 )


Tree_opt = fitrtree(Xcart, Ycart, 'OptimizeHyperparameter', 'auto' );
Ycart_v_opt=predict(Tree_opt,Xcart_v);
Ycart_v_opt = [Fx_v(1); Ycart_v_opt];
y_best_cart = Ycart_v_opt.*sigmaF_v + F_v;
R2_cart_best = 1 - sum( (flow_val( 2 : end ) - y_best_cart( 2 : end )).^2 ) / sum( (flow_val( 2 : end )-F_v(2:end) ).^2 )

%% Final resuls:

% It is clear that linear models are the best ones. By putting an high
% number (40) of "maxOrder" variable, best absolute model has been
% ARX(18,19,19) with R2=0.9804. In order to not overcomplicate the model 
% "maxOrder" variable has been reduced to 5 and best model ARX(4,5,5) gives
% an extraordinary near R2 of 0.9801. 


