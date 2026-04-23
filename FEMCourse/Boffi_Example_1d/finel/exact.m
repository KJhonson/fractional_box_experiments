function value=exact(i,x)
if i==1
  value=x.*(1-x)/2;
elseif i==2
  value=exp(3*x)-4*cos(x);
elseif i==3
  value=1-abs(x).^(5/4);
end
