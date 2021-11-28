function [y, R2] = ARX_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Px,Px_v,Tx,Tx_v,n,m1,m2)

%THIS FUNCTION PERFORM THE COMPUTATION OF ARX MODEL FOR STREAMFLOW FORECAST OF THE
%RIVER GIVEN TIME SERIES OF STREAMFLOW AND  TWO EXOGENOUS SIGNAL, 
%PRECIPITATION AND TEMPERATURE.

%REQUIRED INPUT ARE:

%   1)order of the desired ARX model: n is the order of the AR part, m1 the
%   order of first exogenous part (precipitations) and m2 order of second
%   exogenuos part (temperature).

%   2)training and validation data (deTrendized!)
%   3)cyclostationary mean and variance to rebuild correct signal
%   4)real data for comparison and R2 computation

%IN DETAILS

%flow_val: real time series of the process

%F_v: cyclostationary mean (repeated for the entire length) of the process
%validation data

%sigmaF_v: cyclostationary variance (repeated for the entire length) of the
%process validation data

%Fx: deTrendized times series of process training data
%Fx_v: deTrendized times series of process validation data

%Px: deTrendized times series of first exogenous training data
%Px_v: deTrendized times series of first exogenous validation data

%Tx: deTrendized times sereies of second exogenous training data
%Tx_v: deTrendized times series of second exogenous validation data


%TRAINING
%the harder part is to build the correct  general Y and M matrix of
%training data in order to use least square.

Y=Fx(1+n:end);
M=Fx(1:end-n);
if(n>1)
for i=2:n
    M=[Fx(i:end-(n-i)-1), M];
end
end
if(m1>0)
    for i=1:m1
        M=[M, Px(i:end-(m1-i)-(n+1-m1))];
    end
end
if(m2>0)
    for i=1:m2
        M=[M, Tx(i:end-(m2-i)-(n+1-m2))];
    end
end

%LEAST SQUARE SOLUTION

theta = M\Y;

%VALIDATION

M_v=Fx_v(1:end-n);
if(n>1)
for i=2:n
    M_v=[Fx_v(i:end-(n-i)-1), M_v];
end
end
if(m1>0)
    for i=1:m1
        M_v=[M_v, Px_v(i:end-(m1-i)-(n+1-m1))];
    end
end
if(m2>0)
    for i=1:m2
        M_v=[M_v, Tx_v(i:end-(m2-i)-(n+1-m2))];
    end
end

%OBTAINED TIME SERIES

x = M_v*theta;

%RECONSTRUCTION OF THE SIGNAL

x = [Fx_v(1:n); x];
y = x.*sigmaF_v+F_v;

%PERFORMANCE INDEX COMPUTATION (R SQUARE)

R2 = 1 - ( sum( (flow_val( 1+n:end ) -y(1+n:end)).^2) / sum( (flow_val(1+n:end ) - F_v(1+n:end)).^2) );
end

