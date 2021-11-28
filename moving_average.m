function [ mi , ym ] = moving_average( y , T , f )

% [ mi , ym ] = moving_average( y , T , f )
% 
% input:
% - y  --> time-series
% - T  --> period
% - f  --> semi-amplitude of the window
%
% output:
% - mi --> periodic moving average
% - ym --> periodic moving average repeated for the length of the
%          time-series.
%


N_years  = length( y ) / T           ;
Y        = reshape( y , T , N_years ) ;

Y_       = [ Y( end - f + 1 : end , end ) , Y( end - f + 1 : end , 1 : end - 1 ); ...
            Y                                                                   ; ...
            Y( 1 : f        , 2 : end )  , Y( 1 : f , 1 )                      ];

mi      = zeros( T , 1 )            ;
for k = 1 : T
    mi( k ) = mean( mean( Y_( k : k + 2*f , : ) ) ) ;
end
ym      = repmat( mi , N_years , 1 ) ; 
