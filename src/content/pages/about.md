---
title: "About Niru - Niru ML"
description: "Niru ML is an independent research initiative led by Brendon Reperttang. It is building AMLO, a system designed to do machine-learning research at a fraction of the cost, and to show its work at every step."
headline: "About Niru"
lede: "Niru ML is an independent research initiative led by Brendon Reperttang. It is building AMLO, a system designed to do machine-learning research at a fraction of the cost, and to show its work at every step. This page sets out the bet behind it, where the name comes from, and who is making the bet."
boundary: "Brendon Reperttang operates Niru ML personally, in California, USA. It is not yet a corporation or a registered company."
updated: 2026-10-13
---

## The bet

Model development rests on a bet that is rarely stated. What wins at a size you can afford will still win at the size you care about. Train a thousand candidates small. Rank them. Scale the winner.

The bet can fail, and nothing flags the failure. Under the standard parameterization, a wider network trains differently. Its best learning rate moves, and its features change less as it trains ([Yang and Hu, 2021](https://arxiv.org/abs/2011.14522); [Yang et al., 2022](https://arxiv.org/abs/2203.03466)). A small proxy and its large target can then sit in different regimes, and a ranking found at one size need not hold at the other.

Comparisons at a single size fail too. A candidate that trained a little longer, or drew a better learning rate, looks like a better architecture. The best of a hundred noisy scores is biased upward, because the noise helped to choose it. In each case a result looks like progress because the evidence that would refute it was never collected. Niru calls this a false green.

Niru’s bet is different. With the right abstraction, a small model can keep the training behavior of a large one to a small error, at a cost orders of magnitude lower. Where that holds, research that now needs a large run can be done at small scale. The bet is not taken on faith. The parameterization is chosen so that small and large train alike, the transfer is checked before a large run is paid for, and every decision goes into a record with its reason. AMLO is how Niru tests the bet. The home page sets out [the standard every result is held to](/#standard).

## The arrow

_Niru_ is the Manchu word for an arrow (and also a company, the smallest unit of the Eight Banners that formed the Qing army), and for one kind of arrow in particular, the broadhead. The Manchu bow had a heavy draw.
In the right hands it was the strongest weapon on the field. In other hands it
was only heavy.

The Banners' empire learned late, and then fast. Between 1872 and 1875 the Qing court sent 120 boys to study in the United States. One of them, Zhan Tianyou, came home
an engineer, and in 1909 his railway from Beijing to Zhangjiakou opened, the first major line in China that Chinese engineers designed and built.

The arrow on this site is cut in wood, in the manner of the oldest printed books. The image is a product of what the knife leaves behind.

## The names

_Repert_ is a Germanic name. My relatives reached America, hailing from England, Scotland, Ireland, Scandinavia, and the German-speaking lands along the Rhine. Around 1455, in Mainz on the Rhine, Gutenberg printed a Bible from movable metal type. Sixty years later Erasmus put the Greek New Testament into print, and scholars could read the original for themselves.

_Tang_ is the dynasty that governed China from 618 to 907. Its law code, issued with an official commentary in 653, is the oldest Chinese legal code that survives complete. It has 502 articles, and the commentary beside each one explains the de jure meaning and reasoning behind it. Japan, Korea and Vietnam built their own codes on it.

An edict passed through three departments. One drafted it, one reviewed it and could send it back, and one carried it out. In 868 a man named Wang Jie paid for the Diamond Sutra to be cut into wooden blocks, [“Reverently made for universal free distribution”](https://www.smithsonianmag.com/smart-news/Five-things-to-know-about-diamond-sutra-worlds-oldest-dated-printed-book-180959052/), in memory of his parents. It is the oldest printed book with a date.

## The founder

In March 2016 I watched a program beat Lee Sedol at Go. That match set me towards this field. I had followed deep learning loosely since 2012, when a convnet won the ImageNet contest by a margin that settled [_"the argument"_](https://onlinelibrary.wiley.com/doi/pdf/10.1002/aaai.12036#page=7) for the time being.
In December 2016, in my junior year at Point Loma High, an English class asked
us to present something we had made. Mine was on radial basis function
networks and temporal-difference learning. The transformer paper came out the next summer.

At MIT, by sophomore spring, I had moved from foundations towards time-series forecasting and computer vision.
Well into junior year, I turned to category theory, and with it to the places where mathematics, systems and learning meet. As a senior, I took optimization at Sloan, modeling with machine learning, multi-agent learning and other special subjects as a listener. I changed course to pursue what became Niru ML, graduating in 2024 in Computer Science and Engineering.

I read Rich Sutton’s [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) as a test where a given method must keep improving as compute grows. I believe some families of models stay the best choice at every scale under the correct parameterization. They learn more from less data and compute, and they keep what makes intelligence general while becoming more robust. Finding them is a search problem. AMLO is built to run that search under a protocol that someone else can check.

My conviction is that aligned superintelligence for all, reached quickly and efficiently, needs neither miraculous solutions nor a king’s ransom. Cheaper, checkable research is the part of that path I can test now. For as long as I have worked in this field, I have wanted a system that turns its method on itself and improves the way it improves, with each step of that recursion recorded and open to audit. The hard problems deserve solutions that are efficient as well as scalable, and software around them that is verified, not trusted.

If you are an investor or a possible partner, the best first step is a conversation.
