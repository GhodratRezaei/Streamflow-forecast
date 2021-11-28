function [y, R2] = ANN_solver(flow_val,F_v,sigmaF_v,Fx,Fx_v,Px,Px_v,Tx,Tx_v,n)

% This function performs the modeling throught an artificial neural
% networks. Input of the function are basically the same of ARX_solver, with
% only difference that n is now the number of layers in the ANN. Since Deep
% Learning toolbox works with raws and not vectors, all matrices must be
% transposed. Function performed net training and then simulation on
% validation data.
% Function outputs R2 values and simulated time series of streamflow.

X = [Fx(1:end-1), Px(1:end-1), Tx(1:end-1)]';
Y = Fx(2:end)';  
X_v =[ Fx_v(1:end-1), Px_v(1:end-1), Tx_v(1:end-1)]';

net = newff(X,Y,n);
net = train(net,X,Y);
Y_v = sim(net, X_v );
Y_ann = [Fx_v(1); Y_v'];
y = Y_ann .*sigmaF_v + F_v;
R2 = 1 - sum( (flow_val( 2 : end ) - y( 2 : end )).^2 ) / sum( (flow_val( 2 : end )-F_v(2:end) ).^2 );
end
