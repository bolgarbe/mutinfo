using StatsBase, SpecialFunctions, QuadGK, JLD, Roots, NPZ, SparseArrays, LinearAlgebra
using PGFPlots, TikzPictures

define_color("pastelBlue",[27/255,161/255,234/255])

const A000367 = [1,1,-1,1,-1,5,-691,7,-3617,43867,-174611,854513,-236364091,8553103,-23749461029,8615841276005,-7709321041217,2577687858367,-26315271553053477373,2929993913841559,-261082718496449122051]
const A002445 = [1,6,30,42,30,66,2730,6,510,798,330,138,2730,6,870,14322,510,6,1919190,6,13530]
const bernoulli = A000367 .// A002445

function Ψ(x::BigFloat)
    ψ = zero(x)
    if x < 7
        n = 7 - floor(Int,x)
        for ν in 1:n-1
            ψ -= 1/(x+ν)
        end
        ψ -= 1/x
        x += n
    end
    t = 1/x
    ψ += log(x) - 0.5*t
    t *= t
    ψ -= t * @evalpoly(t,(bernoulli[2:20] .// (2*(1:19)))...)
end

function Ψ1(x::BigFloat)
    ψ = zero(x)
    if x < 8
        n = 8 - floor(Int,x)
        ψ += 1/x^2
        for ν = 1:n-1
            ψ += 1/(x+ν)^2
        end
        x += n
    end
    t = inv(x)
    w = t * t
    ψ += t + 0.5*w
    ψ += t*w * @evalpoly(w,bernoulli[2:20]...)
end

p(α,K) = (K*Ψ1(α*K+1) - Ψ1(α+1))/log(K)
function E(α,n,c,K)
    N = sum(c.*n)
    Ψ(α*K+N+1) - sum(@. c*(α+n)/(α*K+N)*Ψ(α+n+1))-(K-sum(c))*α/(α*K+N)*Ψ(α+1)
end

pBh(α,K1,K2) = K1*Ψ1(α*K1+1) + K2*Ψ1(α*K2+1) - (K1*K2)*Ψ1(α*(K1*K2)+1) - Ψ1(α+1)
tB(α,K1,K2) = Ψ(α*K1+1) + Ψ(α*K2+1) - Ψ(α*(K1*K2)+1) - Ψ(α+1)
function HB(Y1,Y2)
    N1,M1 = size(Y1)
    N2,M2 = size(Y2)
    K1,K2 = BigInt(2)^M1,BigInt(2)^M2
    
    z   = find_zero(a -> pBh(a,K1,K2),(BigFloat(0),BigFloat(10)))
    tBz = tB(z,K1,K2)
    pB(α) = -abs(pBh(α,K1,K2))/2tBz
    
    cnt1  = countmap(values(countmap(eachrow(Y1))))
    cnt2  = countmap(values(countmap(eachrow(Y2))))
    cnt   = countmap(values(countmap(eachrow([Y1 Y2]))))
    n1,c1 = collect(keys(cnt1)),collect(values(cnt1))
    n2,c2 = collect(keys(cnt2)),collect(values(cnt2))
    n,c   = collect(keys(cnt)),collect(values(cnt))
    quadgk(a->pB(a)*(E(a,n1,c1,K1)+E(a,n2,c2,K2)-E(a,n,c,K1+K2)),BigFloat(0),BigFloat(Inf),rtol=1e-8,order=8)[1]
end

Y  = npzread("data/Y.npy")

cids = [108,179,181]
srts = [
    [245,389,866,390,1335,1336,1349,1366,1382,1383,1509,1550,519,1348,1365,520,522,1490],
    [556,1434,554,555,667,668,669,717],
    [406,307,391,306,246,404,321,247,323,366,367,368,322,405]
]

dct = Dict{Int,Vector{BigFloat}}()
for it in 1:3
    dct[cids[it]] = [HB(Y[:,srts[it][1:1]],Y[:,srts[it][2:i]]) for i in 2:length(srts[it])]
end

gp = GroupPlot(3,1,groupStyle="horizontal sep=.6cm, y descriptions at=edge left")
for it in 1:3
    srt = srts[it]
    cid = cids[it]
    res = dct[cid]
    lab = "{" * prod(["$(i-1)," for i in srt[2:end]]) * "}"
    push!(gp,Axis([
        Plots.Linear(1:length(res),res,
            style="solid,thick,pastelBlue,mark=none"),
    ],title="Cluster $cid (core: $(srt[1]-1))",
    xlabel="Auxiliary tasks",
    width="6.5cm",height="6.5cm",
    ymin=0,ymax=log(2),xmin=1,xmax=length(res),
    style="xtick={1,2,...,$(length(res))},xticklabels=$lab,x tick label style={rotate=90}",
    ))
end
gp.axes[1].ylabel="Conditional MI"
PGFPlots.save("mi.pdf",gp)
