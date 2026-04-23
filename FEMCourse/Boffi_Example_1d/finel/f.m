function value=f(i,x)
if i==1
  value=1*ones(size(x));
elseif i==2
  value=-9*exp(3*x)-4*cos(x);
elseif i==3
  value=min(Inf,5/16*abs(x).^(-3/4));
end
