Chapter 1: Intro

0:044 secondsI had long been interested in kind of, what fundamentally is there underneath physics?

0:1010 secondsWhat is the sort of foundational things that create our physical universe? What rules could those be?

0:1818 secondsI at first thought when the rules are simple enough, the behavior one will get will always be correspondingly simple.

0:2424 secondsBut I started doing actual computer experiments to find out what really happens.

0:2727 secondsIt's kind of like you get to take the computer like a telescope, turn it at the sky and see what you see. What I saw was really very remarkable.

0:3737 secondsAt first, I didn't quite believe it.

0:3939 secondsEven when the rules by which you're operating are very simple, the behavior of the system can be very complicated. I'm Stephen Wolfram.

0:4747 secondsI've been motivated for a long time by understanding the significance of the paradigm of computation,

0:5454 secondsboth in its practical applications in technology and its implications for kind of deep conceptual issues in thinking about the world.

Chapter 2: Chapter 1: The limits of theoretical physics

1:081 minute, 8 seconds- Chapter 1: The limits of theoretical physics - The story of 20th century physics was a story of discovering many different phenomena.

1:181 minute, 18 secondsAnd the question was, do all these phenomena fit together? And it hadn't been figured out how that could work.

1:251 minute, 25 secondsYou know, in antiquity, people just sort of said we can think about how the universe works and we can conclude that it's made of atoms or everything flows or something like that.

1:341 minute, 34 secondsIt's kind of a pure, we just think about it to see how it works.

1:381 minute, 38 secondsBy the 1600s, people were starting to use kind of mathematical methods to understand the natural world and so on.

1:441 minute, 44 secondsThey picked out certain aspects of the natural world that were amenable to analysis by mathematical methods. I

1:511 minute, 51 secondsI mean, you know, Newton, for example, was skilled in realizing that he should study mechanics of solid objects.

1:571 minute, 57 secondsYou know, what is the trajectory of this thing? How does it move when acted on by forces?

2:032 minutes, 3 secondsHad he studied fluids, which can behave in much more complicated ways and show turbulent random behavior, he wouldn't have been able to deduce these kind of simple laws of motion to kind of launch the whole Newtonian story about physics.

2:172 minutes, 17 secondsBut there was still at that time, people like, I don't know, Descartes was saying, you know, within 100 years, we'll just have understood everything.

2:262 minutes, 26 secondsWe will have managed to just decode how our universe works. That didn't happen.

2:322 minutes, 32 secondsWhat seemed to happen was, as people discovered more kinds of phenomena, the theories got more complicated.

2:382 minutes, 38 secondsAnd what happened in, well, going into the 20th century, I would say that an important sort of trend was the idea of formalization of things, which came to

2:482 minutes, 48 secondsmathematics in the course of the 19th century, the idea that mathematics could be built up

2:542 minutes, 54 secondsas an almost logical, structural kind of thing, rather than something which was kind of an almost empirical way of describing the world as we perceive it.

3:043 minutes, 4 secondsThat was something which coming into the 20th century, I think, influenced kind of thinking about physics.

3:093 minutes, 9 secondsAnd the three big theories that have sort of dominated 20th century physics are general relativity, the theory of space-time and gravity, quantum mechanics,

3:193 minutes, 19 secondsand statistical mechanics, where sort of the key insight is the second law of thermodynamics, the idea

3:263 minutes, 26 secondsthat there's a tendency for systems to kind of get more random in their behavior through time. Those are sort of the three big theories of 20th century physics.

3:343 minutes, 34 secondsAnd they all come out of, they all rely on this kind of formalization that was a thing that had sort of emerged in the 19th century.

3:423 minutes, 42 secondsYou know, then the question is, okay, so we've got these general ideas about how things work. What specifically, how specifically is our universe built?

3:523 minutes, 52 secondsAnd is there some underlying structure that is underneath those theories?

3:583 minutes, 58 secondsWell, there were ideas like supersymmetry and string theory and so on, a whole collection of ideas which never really connected with the actual experimental observations that had been made.

4:114 minutes, 11 secondsAnd there were still a lot of kind of things that seemed very arbitrary, like, you know, there are electrons and they have a certain mass.

4:184 minutes, 18 secondsAnd then there are muons that are just like electrons, but they're 206 times heavier. Why does that exist?

4:234 minutes, 23 secondsThere were a lot of kind of mysterious, it just happens to be that way kinds of things.

4:284 minutes, 28 secondsAnd there was sort of an attempt for a few decades to make that something that would be more explainable, but didn't work out.

4:354 minutes, 35 secondsMy own efforts really started from a different place.

4:384 minutes, 38 secondsThey started not from looking at kind of the world as it is and trying to reverse engineer using mathematical ideas to see how one could derive that world from mathematical ideas.

4:504 minutes, 50 secondsRather, what I was doing was something in a sense much more pedestrian, which was is to say, let's look at this computational universe of possibilities based on the operation of simple programs.

5:015 minutes, 1 secondLet's see what those things do.

5:035 minutes, 3 secondsAnd maybe somehow we will see that those things behave in ways that are like the way that we observe our physical universe to behave. And the remarkable thing is that worked out.

5:125 minutes, 12 secondsAnd it became clear to me that we're really being able to see kind of what is the fundamental machine code of our universe, what's sort of underneath all the physics we know?

5:225 minutes, 22 secondsIt's been a very exciting thing for me. And in fact, the methods that have come from that understanding of physics have turned out to also have a lot of implications for the foundations of

5:315 minutes, 31 secondsmathematics, foundations of computation, foundations of machine learning, and foundations of biology.

5:375 minutes, 37 secondsThis notion that one can think about things kind of in computational terms seems like a lot of questions that have been around for a century or more, we're starting to be able to unlock.

Chapter 3: Chapter 2: A computational understanding of the world

5:505 minutes, 50 seconds- Chapter 2: A computational understanding of the world - Okay, so what is computation?

5:575 minutes, 57 secondsThe way I see computation, it's the following of rules and seeing what happens.

6:036 minutes, 3 secondsSo we used to computers. Computers have certain built-in machine instructions.

6:106 minutes, 10 secondsWe write programs that apply those machine instructions, and the computational process is seeing the consequences of applying those rules.

6:216 minutes, 21 secondsSo for me, computation at its heart is the consequences of applying rules.

6:296 minutes, 29 secondsAnd those rules can be very simple. They can be more elaborate. That is what computation is, is the application of simple rules.

6:366 minutes, 36 secondsIn fact, for me, the field of study that has to do with looking at simple rules and seeing what their consequences are, I've tended in recent years to call that ruliology, the study of rules and their consequences.

6:496 minutes, 49 secondsComputation is a metaphor that is very useful because we're familiar with computers and how they work.

6:566 minutes, 56 secondsRuliology is really the core basic science about simple rules and how they behave. So those rules might be the rules of arithmetic.

7:067 minutes, 6 secondsAnd the computation might be working out some arithmetic calculation, those rules might be something that says you've got a collection of black and white cells in a

7:157 minutes, 15 secondsline, and you're determining the of the cell as you go down the page, for example, according to a rule that depends on the color of the cells next to that particular cell on the row before.

7:287 minutes, 28 secondsYou state a rule for how things should work, and the computation is the result of applying those rules.

7:357 minutes, 35 secondsSo, for example, when we think about the whole universe, our current picture is that there's this kind of network of atoms of space, and the

7:437 minutes, 43 secondscomputation that the universe is doing is progressively rewriting that network of atoms of space,

7:507 minutes, 50 secondsand doing that by saying, whenever there's a piece of network that looks like this, replace it by one that looks like that, and so on.

7:577 minutes, 57 secondsWhen one's interested in kind of exploring the computational universe, there's kind of a question of what types of computations are most easily understood.

8:068 minutes, 6 secondsAnd the one that I was sort of lucky enough to come upon at the beginning of the 1980s are some things that had, a slightly different version of them had

8:168 minutes, 16 secondsbeen called cellular automata, and I kind of took that historical name. A cellular automaton is a row of cells.

8:238 minutes, 23 secondsEach cell can be either black or white, for example, and the thing progresses, kind of making a picture down the page, line by line, and each successive line is

8:348 minutes, 34 secondsformed from the one above it by taking each cell and asking, what were the colors of that cell and the two neighbors of that cell,

8:418 minutes, 41 secondsfor example, on the step before, and then looking up some table to how that cell should be colored on the next step.

8:478 minutes, 47 secondsThe idea of a cellular automaton could easily have existed in antiquity.

8:518 minutes, 51 secondsAnd I've been kind of waiting for the time when some archaeological artifact will be unearthed that is a cellular automaton image. It's a very simple idea.

9:009 minutesThe remarkable thing about it is just taking those very simple rules and applying them can lead to very complicated patterns, can lead to patterns that are in some sense as complicated as anything can be.

9:129 minutes, 12 secondsThat's the very remarkable discovery and observation that even very simple rules can lead to sort of arbitrarily complex behavior.

9:219 minutes, 21 secondsCellular automata are convenient because they have a very immediate visual aspect. You can kind of see the computation they do.

9:299 minutes, 29 secondsThere are plenty of other models of computation. There are Turing machines invented by Alan Turing in 1936.

9:359 minutes, 35 secondsThose are things where there's kind of a tape with a bunch of ones and zeros, for example, on it. And you have this head that's moving back and forth according to certain rules.

9:459 minutes, 45 secondsThere are things called combinators, which were actually invented in 1920 by a chap called Moses Schönfinkel, they were actually the very first example of a sort of complete computational system, but they're incredibly obscure.

9:589 minutes, 58 secondsAnd even to this day, they're very hard to understand. I wrote a book about combinators recently at the centenary of combinators.

10:0610 minutes, 6 secondsAnd I have to say that they're very hard for us to wrap our minds around.

10:1110 minutes, 11 secondsThen there are kind of approaches like so-called register machines, which are kind of very simple idealization of the way that practical computers work.

10:1910 minutes, 19 secondsThere's a whole collection of these things.

10:2110 minutes, 21 secondsThe remarkable thing is, even though at first, all these different kind of models of computation seem very different.

10:2710 minutes, 27 secondsThey're talking about cells, they're talking about numbers, they're talking about sort of symbolic operations and things. They seem very different.

10:3510 minutes, 35 secondsOne of the remarkable things is that they all, in fact, are equivalent. You can emulate a Turing machine with a cellular automaton.

10:4410 minutes, 44 secondsYou can emulate a cellular automaton with a Turing machine. You can use combinators to emulate a register machine and so on.

10:5010 minutes, 50 secondsIt's something that we have some familiarity with because we know that we can buy different kinds of computers, different kinds of computer hardware, and yet we can run the same software on those different computers.

11:0111 minutes, 1 secondThe details of the program are different, but the operation of the program when we're using it is the same on those different computers. That turns out to be a much more general phenomenon.

11:1011 minutes, 10 secondsSomething that was understood by the 1940s and so on was that these different models of computation had this kind of equivalence.

11:2011 minutes, 20 secondsWhat was not clear is whether that equivalence extended to the physical world.

11:2411 minutes, 24 secondsIt's very easy to just start saying, what are all the possible programs that you can make, let's say, with black and white cells and nearest neighbor rules and things like this.

11:3511 minutes, 35 secondsYou can just enumerate them.

11:3711 minutes, 37 secondsFor cellular automata, the very simplest class, I tended to call elementary cellular automata, there are 256 of those.

11:4411 minutes, 44 secondsAnd over the last 45 years or so, pretty much every one of those 256 rules has ended up being useful as a model of something.

11:5311 minutes, 53 secondsIt's kind of a strange thing. what happens when you get sort of very fundamental kinds of models. They end up having applications in lots of different areas.

Chapter 4: Rule 30: a simple program that outputs pure randomness

12:0312 minutes, 3 secondsMy kind of all-time favorite is Rule 30, which has the feature that it's just that all of these rules are very tiny and very simple.

12:1112 minutes, 11 secondsThe numbers kind of just come from taking the number 30 and writing it out in binary, and that's a representation of what this rule actually does.

12:2112 minutes, 21 secondsIn terms of black and white cells and so on. Well, Rule 30 has this feature that you start it off from just one cell and it produces this extremely complicated pattern.

12:3112 minutes, 31 secondsSome aspects of the pattern, it's kind of regular on one side, but if you look, for example, at the center column of this triangular kind of pattern, the center column, for all practical purposes, looks completely random.

12:4212 minutes, 42 secondsYou can't predict it in any way.

12:4412 minutes, 44 secondsThe only way you can find out what it's going to be, it seems, is by running Rule 30 and seeing what happens. kind of a little bit like what you get in something like the digits of pi.

12:5412 minutes, 54 secondsIt's easy to say what pi is. It's the ratio of the circumference diameter of a circle.

12:5912 minutes, 59 secondsThere are methods for computing pi, but once you've computed it, you know, 3.14159, etc., the digits of pi look completely random.

13:0913 minutes, 9 secondsAnd it's the same kind of thing with Rule 30, except Rule 30 is a much simpler kind of setup and one that is much closer to the kinds of things that you can expect to see in the natural world.

13:1913 minutes, 19 secondsSo what are some examples of natural systems that kind of can be described easily in terms of programs?

13:2913 minutes, 29 secondsOne example, snowflake growth. Snowflakes grow by aggregating pieces of ice onto a growing structure.

13:3713 minutes, 37 secondsAnd there are very simple rules that describe the way in which a snowflake grows arms and then the arms grow arms and so on.

13:4513 minutes, 45 secondsThat's something that is very easy to describe in simple computational terms.

13:4913 minutes, 49 secondsAnother example, in biology, lots of growth processes are fundamentally computationally quite simple. Whether it's

13:5713 minutes, 57 secondsthe way that a mollusk shell kind of grows in a spiral, or whether it's the way the pigmentation on mollusk shells work, whether it's kind of

14:0514 minutes, 5 secondsjust a row of cells and they're each either producing pigment or not, and it's actually exactly

14:1214 minutes, 12 secondsone of these cellular automata where you have either you get the pigment or you don't get the pigment in each cell at each step and the result is

14:2014 minutes, 20 secondsthis kind of elaborate pattern on the shell of the mollusk Biology has traditionally not been a theoretical science.

14:2814 minutes, 28 secondsBiology has been a science where you just observe, this is how organisms work, this is how molecular biology works, and so on.

14:3614 minutes, 36 secondsIt's not been something where one anticipates that there will be some theoretical basis. Really, there are basically two basic theories in biology.

14:4514 minutes, 45 secondsOne is natural natural selection, theory of biological evolution, and the other is kind of the molecular nature of biology and particularly the nature of DNA and so on.

14:5714 minutes, 57 secondsThose have been kind of all we've had in terms of thinking about biology sort of at a theoretical level.

15:0215 minutes, 2 secondsNow, when it comes to even biological evolution, It's not been completely clear why it works. Why doesn't it get stuck?

15:0915 minutes, 9 secondsWhy can biological evolution go on and produce more and more elaborate forms?

15:1415 minutes, 14 secondsYou know, Darwin, I think, believed, I think the last sentence of Origin of Species talks about how, you know, as the Earth goes circling around the sun,

15:2115 minutes, 21 secondsaccording to the fixed law of gravity, so sort of more and more complex organisms are being produced.

15:2715 minutes, 27 secondsHe thought that there would be some kind of abstract law or kind of law like the laws of physics that would determine the progress of biological evolution.

15:3815 minutes, 38 secondsBut nobody found that.

15:4015 minutes, 40 secondsI came back to that a couple of years ago and tried to understand sort of in computational terms, how does biological evolution work?

Chapter 5: Evolution and machine learning are the same trick

15:4815 minutes, 48 secondsAnd I had one very important new data point, which was machine learning and the fact that machine learning works.

15:5415 minutes, 54 secondsYou know, I had played around the neural nets and kind of the foundations of machine learning back in the early 1980s. I'd never managed to get neural nets to do anything interesting.

16:0316 minutes, 3 secondsWhat was discovered in the 2010s was the surprising fact that if you take a neural net and you just bash it really, really hard, eventually it will learn stuff.

16:1216 minutes, 12 secondsAnd so I thought, well, let's try that same idea. for thinking about biological evolution.

16:1816 minutes, 18 secondsBiological evolution and machine learning turn out to be extremely related kinds of things. And so I did try that.

16:2416 minutes, 24 secondsAnd to my considerable surprise, yes, these simple computational systems could kind of evolve to get more and more complex forms to achieve more and more elaborate fitness objectives and so on.

16:3816 minutes, 38 secondsOne of the things that is always remarkable about studying sort of rules, systems out there in the computational universe, it's a very

16:4616 minutes, 46 secondshumbling experience because you typically have some hypothesis about what you're going to see.

16:5116 minutes, 51 secondsYou do some big search for things, you explore things, you kind of are observing things through your computational telescope, and always the things do something different than you expected.

17:0217 minutes, 2 secondsYes, it really is the case that there's this kind of thing that is probably the secret that nature uses to make all this complexity that it makes, which is that in

17:1117 minutes, 11 secondsthe computational universe of possible programs, even a very simple program can produce extremely complicated behavior.

17:1817 minutes, 18 secondsAnd that kind of led me to this concept of computational irreducibility that's been an important one for a lot of what I've done.

17:2517 minutes, 25 secondsKind of this question of when you have a simple program, if you want to know what it does, one thing you can do is just run it step by step.

17:3217 minutes, 32 secondsLet's say you want to know what it's going to do after a million steps, where you can run those million steps and see what it does.

17:3717 minutes, 37 secondsThe question is, can you jump ahead and work out what it's going to do without having to go through all those steps?

17:4317 minutes, 43 secondsOne had had the view that came from kind of the success of mathematical science and physics and so on, that in some sense, everything was computationally reducible.

17:5417 minutes, 54 secondsWhatever was going on in the world, you would be able to sort of be smarter than those things and say, I know how it's going to work out, predict what's going to happen.

18:0318 minutes, 3 secondsAnd that's the thing, you know, in the tradition of the exact sciences, the idea that you can predict things, that you can say, I don't need to follow a million orbits of the Earth around the sun or some idealized sun.

18:1518 minutes, 15 secondsI can just use a formula and jump ahead and say what's going to happen.

Chapter 6: What computational irreducibility means for science

18:1918 minutes, 19 secondsOne of the things that has come out of what I've studied that is the result of this idea of computational irreducibility is that, no, science isn't that powerful.

18:2918 minutes, 29 secondsScience is not able to make statements, in many cases, about what will happen.

18:3518 minutes, 35 secondsThe only way to find out what will happen is basically just to run the system and see what happens. It's a fundamental limitation of science that comes from within science.

18:4418 minutes, 44 secondsI mean, kind of in historical lineage, it's kind of a descendant of things like Gödel's Theorem in mathematics, but it's, more directly it's a consequence of the principle of computational equivalence and computational irreducibility.

18:5718 minutes, 57 secondsSo that's been one of the things that has been sort of an important thing for me to understand, that there are sort of certain fundamental limitations to the traditional scientific paradigm.

19:0819 minutes, 8 secondsOne of the questions in biology is what's special about life?

19:1419 minutes, 14 secondsEven if you just take a piece of living tissue. What kind of a thing is that? Is a piece of living tissue liquid?

19:2219 minutes, 22 secondsWell, it's kind of gooey often. Is it solid? Well, it's kind of has some, you know, maintains its structure in some way.

19:2919 minutes, 29 secondsIt's really not those things.

19:3119 minutes, 31 secondsWhen you look at it microscopically, so that the big discovery of molecular biology, I suppose, in the last few decades has been that things are very orchestrated.

19:4019 minutes, 40 secondsMolecules, are sort of specifically and actively transported from here to there. This thing fits exactly into that, which then opens up to do this and so on.

19:5019 minutes, 50 secondsThere's this question of how are all these pieces kind of orchestrated together to do the things that happen in biology?

19:5819 minutes, 58 secondsAnd this notion of sort of bulk orchestration that we're full of tons of molecules, but they're all doing things in this very kind of orchestrated way.

20:0720 minutes, 7 secondsThat's a phenomenon that seems to be sort of an essential phenomenon of life.

20:1220 minutes, 12 secondsIt's sort of the result of this big technology stack that's been built up through the course of biological evolution.

20:1820 minutes, 18 secondsIt's one where the pieces are sort of like these kind of lumps of computational irreducibility that have been fit together to make, in the

20:2820 minutes, 28 secondsend, something which achieves the purposes that biological organisms achieve.

20:3320 minutes, 33 secondsThe analogy is the organism is trying to do something like build a wall.

20:3920 minutes, 39 secondsWell, if we were doing that by engineering, we would make bricks that are nice shapes and we would arrange them in some simple pattern.

20:4520 minutes, 45 secondsBut what's happening in biology, and by the way, also in machine learning, is that one is picking up these kind of random lumps of irreducible computation.

20:5320 minutes, 53 secondsThey're kind of like rocks lying around on the ground.

20:5620 minutes, 56 secondsAnd one's fitting those in and saying, well, this one happens to fit this way and this way.

20:5920 minutes, 59 secondsAnd eventually you build up this wall. And the reason that biological evolution works is that the fitness objectives that

21:0721 minutes, 7 secondsexist for biological organisms are computationally very simple compared to sort of the power of this underlying irreducible computation.

21:1421 minutes, 14 secondsI mean, it's sort of unsurprising if every organism, as soon as it was born, had to be able to solve some elaborate mathematical problem, no organisms would survive.

21:2221 minutes, 22 secondsIt's because the sort of fitness objectives are computationally quite simple, particularly relative to sort of the power of these underlying computational elements, that biological evolution is possible and can work, so to speak.

21:3621 minutes, 36 secondsIf we look at the history of science and thinking in general, a lot of that history has to do with formalizing things.

21:4521 minutes, 45 secondsFrom the earliest days of human natural language, It's kind of like one could just be pointing at different kinds of things, but then one sort of formalized the idea of a rock.

21:5621 minutes, 56 secondsAnd then one just has to say the word rock, and one knows that corresponds to any of those things that we might have explicitly pointed at before.

22:0322 minutes, 3 secondsAnd then we get things like logic, which are another sort of formalization in that case of the structure of arguments.

22:1022 minutes, 10 secondsAnd then later on, we get mathematics, which is a kind of a formalization of certain aspects of how the world works or certain aspects of how we think about things abstractly.

22:1922 minutes, 19 secondsComputation is another formalization, is a broader formalization of things in the world and abstract things.

22:2922 minutes, 29 secondssort of the great paradigm of the 21st century, is how do we think about things in computational terms?

22:3622 minutes, 36 secondsIt's a way of structuring our thinking.

22:3822 minutes, 38 secondsIt's a way of allowing us to kind of build towers of consequences because we have something that is formal and structured.

22:4522 minutes, 45 secondsAnd one of the things that's been a big activity of my life is trying to see, how do we represent different kinds of things in the world in computational terms?

22:5622 minutes, 56 secondsHow do we eventually have a computational language for describing the world?

23:0023 minutesThat's what technology I built, Wolfram Language and so on, is all about, is finding computational ways to describe the world.

23:0823 minutes, 8 secondsNow you say, well, what can we not describe in computational terms? It's a good question.

23:1223 minutes, 12 secondsIt's, you know, there keep on being things where I'm like, I don't know how this is going to work.

23:1823 minutes, 18 secondsAnd then you start looking at it and you realize, well, actually, this is something you can very much describe in computational terms, and it's very useful to do that.

23:2623 minutes, 26 secondsSo at this point, I think it's fair to say this is sort of the paradigm that allows one to formalize things.

23:3323 minutes, 33 secondsThis idea that things operate according to rules, and then you're looking at their consequence, that's a very general and very powerful idea.

23:4023 minutes, 40 secondsI mean, if you ask sort of what are the bigger sort of philosophical consequences of this and of knowing, for example, that our whole universe is kind of

23:4823 minutes, 48 secondscomputational all the way down, what difference does it make to sort of in common life that we now believe

23:5523 minutes, 55 secondsthat sort of we can describe the universe even at its most fundamental level in computational terms?

24:0124 minutes, 1 secondI think sort of an analogy to what we're seeing now is kind of what happened in sort of the Copernican story, where the fact that, you know, one had believed.

24:1124 minutes, 11 secondsthat one could understand the world just by common sense.

24:1524 minutes, 15 secondsIt's obvious the earth is standing still and the sun is going around in the sky, so to speak, because that's what we feel.

24:2224 minutes, 22 secondsAnd then what became clear was, well, actually, the math says you can think about it differently. And then it's like, trust the math.

24:3024 minutes, 30 secondsDon't trust your everyday senses. That was kind of, in a sense, the lesson of the Copernican Revolution.

24:3624 minutes, 36 secondsI think what we're seeing now is this idea that, you know, you can really think about things in a structured computational way.

24:4524 minutes, 45 secondsThat's the thing we're learning. And sort of things are computational all the way down.

24:4924 minutes, 49 secondsSo it makes sense to think about things using this kind of way of structuring one's thinking in terms of computation.

Chapter 7: Chapter 3: A new kind of theory of everything

25:0025 minutes- Chapter 3: A new kind of theory of everything - When we start thinking about the world in computational terms, there's a question of sort of what does that mean?

25:1225 minutes, 12 secondsPeople will say things like, if the world is computational, where's the computer that it's running on? is a confusion.

25:1925 minutes, 19 secondsModels are a way of representing what the natural world does. They're not mechanistically what the natural world is doing.

25:2725 minutes, 27 secondsWhen we imagine that there's an equation that governs the motion of the Earth around the sun, for example, we're not imagining that inside the Earth

25:3525 minutes, 35 secondsthere's a little piece of software of the kind that I build, so to speak, that's calculating those mathematical equations.

25:4225 minutes, 42 secondsIt's just the mathematical equations are a way of describing the way the natural world works.

25:4825 minutes, 48 secondsAnd so it is, as we start thinking about sort of a computational model of what is underneath physics, that is a representation of what is happening in the world.

25:5825 minutes, 58 secondsIt's not that the machinery of the world is set up to work the way that we imagine a computation should be done.

26:0626 minutes, 6 secondsSo at a more fundamental level, you can ask questions like, what goes from sort of an abstract model of the world to the actuality of the world?

26:1726 minutes, 17 secondsThat's a complicated philosophical question. It's something people have been asking for a long time. You know, why does the universe exist?

26:2626 minutes, 26 secondsYou know, there are versions of this, I think Spinoza had some line, you know, the universe is the thoughts of God actualized.

26:3326 minutes, 33 secondsThat was sort of a version of how to think about that question. Now, I think we have a very interesting way of thinking about that question.

26:4226 minutes, 42 secondsThis is a deep rabbit hole, but let's go into it. Let's start off with physics. And what is the physical world made of?

26:5026 minutes, 50 secondsWe start back in antiquity, people arguing is the universe discrete or continuous.

26:5526 minutes, 55 secondsWe learn by the end of the 19th century that matter is discrete, kind of the realization that we've had in recent times,

27:0327 minutes, 3 secondsthat's sort of the foundation of the things I've done, is that we can think of space as also discrete. And so we think, what's in the universe?

27:1027 minutes, 10 secondsWhat's the universe made of? Well, we think of it as a bunch of discrete atoms of space. They're not atoms in the sense of chemical atoms.

27:1827 minutes, 18 secondsThey're atoms in the sense of being indivisible units.

27:2227 minutes, 22 secondsThere are atoms of space, and the only thing one can say about the atoms of space are how they are related to each other.

27:2927 minutes, 29 secondsAnd we can represent that by a network where we say this atom of space is kind of connected, related to these other atoms. of space.

27:3727 minutes, 37 secondsSo we represent the whole universe just as a graph, a network.

27:4127 minutes, 41 secondsAnd we imagine that the universe and space and everything in it is just features of that network.

27:5027 minutes, 50 secondsWhat there ultimately is, is just this network that represents space and everything in it. Well, another question is, well, what does this network do?

27:5927 minutes, 59 secondsYou know, that might be in sort of technological terms, the data structure of the universe. But what now is the algorithm of the universe, so to speak?

28:0828 minutes, 8 secondsAnd for that, what we imagine is that this network, we look at little pieces of this

28:1528 minutes, 15 secondsnetwork and we say there are a collection of rules that say whenever you see a little piece of network that looks like this, it gets rewritten to a piece of network that looks like that.

28:2528 minutes, 25 secondsAnd this just keeps on happening. And that process of the rewriting of the network, that is the progress of time.

28:3228 minutes, 32 secondsThat is, the time is kind of corresponds to this progressive computational process of the rewriting of this network.

28:3928 minutes, 39 secondsAnd one thing one can ask is, well, what's the large-scale effect of that?

28:4328 minutes, 43 secondsIf there are, you know, 10 to the 400 of these atoms of space, and they're all getting rewritten in all these ways, what does that do in the aggregate?

28:5128 minutes, 51 secondsAnd sort of an analogous situation is what happens in a fluid, where we know what happens at the level of individual molecules colliding and bouncing off each other and so on.

29:0029 minutesBut then the question is, what is the aggregate effect of that when we look at zillions of molecules together?

29:0729 minutes, 7 secondsAnd we know in that case that what emerges is fluid mechanics.

29:1129 minutes, 11 secondsWell, in the case of these networks and atoms of space and so on, what seems to

29:1729 minutes, 17 secondsemerge is general relativity, the theory of gravity, Einstein's equations, and so on.

29:2229 minutes, 22 secondsThat's the thing that is the aggregate effect of all these microscopic processes associated with the structure of this network.

29:3229 minutes, 32 secondsWell, then what happens is there are all these different sort of ways that the network can get rewritten, but there are all these different parts of the network that can be rewritten separately.

29:4229 minutes, 42 secondsThere isn't sort of a single thread of history that says the network is in this form and then in this form and then in this form.

29:4829 minutes, 48 secondsThere are lots of different possible threads of history that correspond to different possible orders in which these little pieces of rewriting can be done.

29:5629 minutes, 56 secondsAnd that possibility of all these different paths of history is what leads to quantum mechanics in our models.

30:0230 minutes, 2 secondsWhat's characteristic for quantum mechanics is that whereas in kind of classical physics, the notion is sort of definite things happen, things follow definite trajectories.

30:1230 minutes, 12 secondsIn quantum mechanics, it's like there are many, many paths that are followed, and we only get to be able to look at sort of the aggregate effect represented in terms of probabilities of all those paths.

30:2330 minutes, 23 secondsSo kind of the picture here is what there ultimately is in the universe is this network of atoms of space.

30:3130 minutes, 31 secondsAnd there are many different sort of paths of history of those atoms of that network. And that's what kind of leads to physics as we know it.

30:4130 minutes, 41 secondsAnd we're getting sort of more and more detail about how physics as we know it emerges from that very simple underlying structure.

30:4930 minutes, 49 secondsBut there's one thing that confused me for a long time, which is with this picture, it's like there's a particular rule for updating this graph, this network and so on.

31:0031 minutesWhy did our universe get that particular rule and not another one. How do we understand that? And of all the infinitely many possible rules, why do we get this particular one?

31:1031 minutes, 10 secondsAnd what I realized in the end is actually we didn't get a particular one. Actually, all possible rules are being used.

31:1631 minutes, 16 secondsWhat's happening is just as there are different paths of history, associated with the different applications of a particular rule, so there are different parts of history associated with the application of different rules.

31:2731 minutes, 27 secondsAnd in the end, the way to think about things is that what one has is this object that represents all possible computations.

Chapter 8: The ruliad: every possible computation, in one object

31:3631 minutes, 36 secondsI call it the ruliad. What is the ruliad?

31:3931 minutes, 39 secondsIt is the entangled limit of all possible computations. The ruliad is a very abstract thing. It's a unique thing.

31:4731 minutes, 47 secondsImagine that you have all possible machines that can do computation, all possible abstract systems that can do computation. You start them all running.

31:5631 minutes, 56 secondsThey are entangled because two different machines may produce the same result.

32:0132 minutes, 1 secondAnd that kind of weaves together the different pieces, the structure of the ruliad. So the ruliad is this very abstract thing.

32:1032 minutes, 10 secondsYou can imagine a notion of rulial space.

32:1332 minutes, 13 secondsSo as you move around the ruliad, it's as if you are using different kind of computational devices to figure out what will happen in the world.

32:2532 minutes, 25 secondsAnd in a sense, you can even think that different minds are embedded in different places in the ruliad.

32:3132 minutes, 31 secondsDifferent minds have a different way of thinking about what's going to happen in the world.

32:3632 minutes, 36 secondsAnd you can represent that by saying those different minds are at different places in the ruliad. So minds that are very closely aligned will be close together in rulial space.

32:4632 minutes, 46 secondsYou know, human minds might be all clumped together. You know, cats and dogs might be a bit further away. The weather with its mind of its own might be much further away.

32:5632 minutes, 56 secondsIt's similar to physical space where we would have a different point of view about kind of what we see out there in the world.

33:0333 minutes, 3 secondsIf we're standing very close together, we'll say we see the same things. If we're standing far apart, we'll say we see different things.

33:1033 minutes, 10 secondsThere actually, in our model of physics, there actually are three different kinds of space that turn out to be important.

33:1633 minutes, 16 secondsPhysical space is the kind we're used to experiencing, what we call branchial space, which is kind of the space of possible histories that's associated with

33:2433 minutes, 24 secondsquantum mechanics, and rulial space, which is this much more general thing that is these kind of different points of view about how the universe works.

33:3333 minutes, 33 secondsBut the ruliad is a completely inevitable, necessary object.

33:3733 minutes, 37 secondsGiven the idea of computation, there is no choice but to have the ruliad The ruliad is this limit of all possible computations.

33:4633 minutes, 46 secondsIt's a unique, inevitable object. So then the question is, well, how do we fit into that?

33:5233 minutes, 52 secondsWe are observers embedded within the ruliad made of the same stuff as the ruliad. And the question is, what is our perception of the ruliad given that setup?

34:0234 minutes, 2 secondsAnd what turns out to be crucial is that we are observers of a certain kind, and observers of the kind we are necessarily perceive the ruliad in certain ways.

34:1334 minutes, 13 secondsLet me try and give a simpler example first.

34:1634 minutes, 16 secondsWhen we're looking at molecules bouncing around, we can ask the question, what do you perceive about these molecules bouncing around?

34:2634 minutes, 26 secondsOne feature of molecular processes is they're reversible.

34:2934 minutes, 29 secondsIf you have a movie of molecules colliding, bouncing off, and so on, you can't tell whether the movie is being run in the forward direction or in reverse.

34:3834 minutes, 38 secondsThe microscopic collisions all look the same.

34:4134 minutes, 41 secondsBut yet, macroscopically, we know you smash a piece of glass, and what you get is something very different.

34:4934 minutes, 49 secondsIt's a very different thing that you don't go backwards from that. We see the thing smashing. We don't see the thing assembling itself spontaneously from all the fragments.

34:5934 minutes, 59 secondsAnd so that's the phenomenon of irreversibility.

Chapter 9: The second law, explained by the limits of our minds

35:0335 minutes, 3 secondsThat's the phenomenon of law of entropy increase, the second law of thermodynamics, and so on. Why does it happen?

35:0935 minutes, 9 secondsWell, it happens actually because of this phenomenon of computational irreducibility.

35:1335 minutes, 13 secondsWhat's happening is the original setup of the system is sort of being run forwards computationally.

35:1935 minutes, 19 secondsThat computation is effectively encrypting whatever simplicity there was in the initial conditions, in the initial setup of the system.

35:2735 minutes, 27 secondsAnd then the issue is, well, what do we see from that initial setup? In principle, we can take whatever comes out and we could reverse it.

35:3535 minutes, 35 secondsBut in practice, because we are observers who are computationally bounded, we can't

35:4135 minutes, 41 secondsdo all that irreversible, irreducible computation to go and follow through all those steps. We're stuck just saying it looks random to us.

35:5035 minutes, 50 secondsIf we could do sort of unbounded computation, we could always know this

35:5635 minutes, 56 secondsparticular elaborate configuration of molecules, that came from the simple initial condition.

36:0136 minutes, 1 secondBut because we are computationally bounded, we have to just say it looks random to us, and we believe in the second law of thermodynamics.

36:1036 minutes, 10 secondsSo the second law of thermodynamics is a consequence of our computational simplicity relative to the computational irreducibility of the underlying processes that are going on.

36:2036 minutes, 20 secondsSo that's an example of a place where the nature of us as observers determines essentially the laws of physics that we perceive.

36:2836 minutes, 28 secondsAnd what seems to be the case is that that's actually a much more general phenomenon that in fact within the ruliad it is the case that observers like us, what does it mean to be like us?

36:3836 minutes, 38 secondsWell, the two characteristics I know for sure are being computationally bounded, having finite minds, not being able to do arbitrarily elaborate computation, being computationally bounded and believing that we are persistent in time.

36:5336 minutes, 53 secondsIt is a surprising thing.

36:5536 minutes, 55 secondsGiven that in these models, we're made of different atoms of space at every successive moment in time, yet we have the internal perception that we experience, we have a thread of experience that's persistent.

37:0837 minutes, 8 secondsAnd so that belief that we have about how things work, that belief and persistence

37:1437 minutes, 14 secondsthat sort of maintenance of the single thread of experience combined with computational boundedness, those two characteristics of us as observers seem to

37:2437 minutes, 24 secondsbe telling us what we have to perceived within the ruliad.

37:2837 minutes, 28 secondsThe ruliad is full of irreducible computation, but it also, one feature of irreducible computation is within irreducible computation, there are always an infinite number of pockets of computational reducibility.

37:4037 minutes, 40 secondsIn other words, even though you can't say everything about what's going to happen, you can always say a few specific things about what's going to happen.

37:4737 minutes, 47 secondsIn fact, there's no limit to the number of specific things you can say about what will happen.

37:5237 minutes, 52 secondsAnd so what we're effectively doing is observers like us kind of are sampling particular pockets of reducibility.

38:0138 minutes, 1 secondI have to say that if everything was purely irreducible, we wouldn't believe there were laws of nature. would just say everything about the universe is unpredictable.

38:1138 minutes, 11 secondsWe just have to watch the universe unfold to see what's going to happen. But in fact, we know there are certain pieces of computational reducibility.

38:1938 minutes, 19 secondsThere are places in which we know there are regularities in the universe, which we, with our finite minds, are capable of making use of

38:2738 minutes, 27 secondsto be able to say that's predictable. That's a law of nature. That's something that allows us to kind of reduce the complexity of the world to something that we can kind of tell a narrative about in our minds.

Chapter 10: Why the universe exists isn't the real question — why we do is

38:3838 minutes, 38 secondsSo the question then is not why does the ruliad exist? The ruliad inevitably exists.

38:4338 minutes, 43 secondsIt's an abstract thing that necessarily exists and is unique. The question that is less obvious is why do we exist?

38:5238 minutes, 52 secondsWhy are there observers like us embedded within the ruliad?

38:5638 minutes, 56 secondsI think that is something which we are within sight of being able to sort of derive scientifically.

39:0139 minutes, 1 secondmean, one feature of observers like us is that we tend to take huge amounts of input data. You You know, we're looking around the room and we're seeing all these pixels of data.

39:1339 minutes, 13 secondsAnd yet we take all that input and we decide we're going to say this word next. We're going to reduce all of that input. We're going to kind of crush it down.

39:2239 minutes, 22 secondsto at a very slow rate decide what we do next. That seems to be a feature of our brains.

39:2739 minutes, 27 secondsIt seems to be an essential feature of the thing that we perceive as consciousness, so to speak, that we are getting the sort of single thread of experience that emerges from the sort of crushing down of all of this input.

39:4039 minutes, 40 secondsIt's something that is different from the rest of the natural world.

39:4339 minutes, 43 secondsI mean, in my principle of computational equivalence, It's not that brains are any more computationally sophisticated than random things in the world.

39:5339 minutes, 53 secondsYou know, people would say sort of whimsically, the weather has a mind of its own.

39:5839 minutes, 58 secondsAnd the principle of computational equivalence says, actually, yes, all those fluid motions in the atmosphere, they're doing a computation that's just as

40:0540 minutes, 5 secondssophisticated as the computation that happens with all the electrochemistry of neurons and brains.

40:1140 minutes, 11 secondsBut the issue is that computation that's going on in the weather is very different in character.

40:1740 minutes, 17 secondsIt's not aligned with the kinds of computations that we do in our brains.

40:2240 minutes, 22 secondsAnd it doesn't have the same feature of taking sort of all of its input data and crushing it down to a single sort of consensus next action.

40:3140 minutes, 31 secondsIt seems that that, sort of, that feature is something that's very essential to our particular way of perceiving the universe. And it's relevant to what we perceive as the laws of physics.

40:4240 minutes, 42 secondsThere are other features of this, like, for example, why do we believe in objective reality?

40:4740 minutes, 47 secondsYou know, each one of us has an internal view of how things work and how we're thinking about things.

40:5340 minutes, 53 secondsBut we believe that there is an outside world there that everybody sort of more or less agrees how it's set up.

41:0041 minutesAnd I think what is surprising to me as well, that it seems like the emergence of sort of a reasonable notion of objective reality depends on the fact that there are lots of us.

41:1041 minutes, 10 secondsIf there was just one of us, wouldn't have a clear notion of objective reality.

41:1441 minutes, 14 secondsIt's because we can all extrapolate that our internal perceptions and feelings are similar in other people.

41:2141 minutes, 21 secondsAnd those other people, we're all observing, we're all sort of agreeing about how the universe, how the world works. And that's why we sort of believe in objective reality.

41:3041 minutes, 30 secondsIt's sort of an interesting feature of that extends to lots of different things.

41:3541 minutes, 35 secondsQuantum mechanics, for example, is a place where it's been very confusing to understand why people think definite things happen in quantum mechanics.

41:4441 minutes, 44 secondsIn quantum mechanics, you're always following these many possible paths of history.

41:4841 minutes, 48 secondsWhy is it, then, people sort of say, yes, a definite thing happened and we agree about what that was?

41:5341 minutes, 53 secondsI think the answer to that is it's because, in a sense, we're all very close together in this thing I call branchial space, the space of possible quantum branches.

42:0342 minutes, 3 secondsIt's similar to physical space. I mean, if you say, what's the night sky like? We'll all say, well, it looks roughly like this. It has these constellations in it and so on.

42:1242 minutes, 12 secondsBut we all say, we all agree about what the night sky looks like because we're all sitting on this one planet. If we were spread throughout the galaxy, we absolutely would not agree.

42:2242 minutes, 22 secondswhat the night sky was like.

42:2342 minutes, 23 secondsIt's because we're sort of close together, we're many entities close together, that we have certain kinds of perceptions about the way the world works.

42:3142 minutes, 31 secondsBut you can kind of keep going and try to understand the extent to which because we are the way we are, then it becomes inevitable that

42:3942 minutes, 39 secondsthe science that we believe in, the laws of physics that we believe in, must be the way that they are, which is sort of a very surprising sort of metaphysical conclusion that I, for one, did not see coming at all.

Chapter 11: Chapter 4: If the universe is a program, what is the meaning of life?

42:5342 minutes, 53 seconds- Chapter 4: If the universe is a program, what is the meaning of life?

42:5842 minutes, 58 seconds- I've been talking a bunch about abstract foundational ideas about how the world is put together and so on. What does that mean for each individual one of us?

43:0643 minutes, 6 secondsI think for me, the key takeaway is foundational knowledge is possible. That is, you can think foundationally about things.

43:1543 minutes, 15 secondsIt's not obvious that you could. It might just be the world is the way it is. We'll never understand it. We'll never be able to think about the foundations of it. But we realize that we can.

43:2343 minutes, 23 secondsWe can drill down and really get to the foundations of things.

43:2643 minutes, 26 secondsAnd certainly in my own sort of experience of life and discovering things and running companies and doing all those kinds of things.

43:3643 minutes, 36 secondsFoundational thinking has been a very key element of the way that I've approached things, that you can understand stuff.

43:4443 minutes, 44 secondsYou can drill down and know what's really going on, know the primitives of what's happening.

43:5043 minutes, 50 secondsNow, the thing about the universe, it's like, how awesome is the universe, so to speak?

43:5643 minutes, 56 secondsAnd if we know the fundamental rules for the universe, does that mean the universe can't be awesome anymore? Does that mean we just got it?

44:0744 minutes, 7 secondsIt's all over. The reason that, in fact, the universe can still be and necessarily will be sort of

44:1544 minutes, 15 secondsfundamentally awesome is this phenomenon of computational irreducibility.

44:1844 minutes, 18 secondsSo in a sense, you might say computational irreducibility is a bad thing because it limits what we can do in science.

44:2644 minutes, 26 secondsIt limits the extent to which we can predict what's going to happen.

44:2944 minutes, 29 secondsBut in some sense, computational irreducibility is what gives us any richness in life.

44:3444 minutes, 34 secondsBecause if we could always predict what was going to happen, nothing is sort of achieved by the actual passage of time and the actual living of our lives.

44:4144 minutes, 41 secondsThe fact that there's computational irreducibility means that sort of the living of our lives adds up to something. We are executing that irreducible computation.

44:5044 minutes, 50 secondsThere's no way that something can come from the outside and say, I know what's going to happen there. It's like you have to live the life to know what's going to happen, so to speak.

Chapter 12: Free will as a side effect of computational irreducibility

44:5944 minutes, 59 secondsComputational irreducibility has a ton of consequences, but one very direct one for us has to do with kind of our perception of free will.

45:0945 minutes, 9 secondsOne might imagine that underneath there are definite rules that govern our behavior, that describe how, you know, the electrical signals in our nerve cells will operate and things like this.

45:2045 minutes, 20 secondsGiven that there are those definite deterministic underlying rules, how can it be the case that the things we do somehow are the result of some kind of free will?

45:3145 minutes, 31 secondsWell, computational irreducibility kind of explains that because what it says is, yes, you know those underlying rules, but to know what the whole critter is going to do, you have to just follow those rules and see what happens.

45:4345 minutes, 43 secondsYou can't say, oh, I know those rules, so therefore I know. the critter is going to stick its head up at just this moment or something.

45:5145 minutes, 51 secondsThat is the thing.

45:5245 minutes, 52 secondsComputational irreducibility is what provides a sort of irreducible gap between the underlying deterministic rules and the actual behavior of a system.

46:0246 minutes, 2 secondsThere will always be new things to discover.

46:0446 minutes, 4 secondsWithin computational irreducibility, there are always an infinite number of pockets of reducibility, and each one of those represents some discovery about how things work.

46:1446 minutes, 14 secondsEach one of those represents a surprise.

46:1646 minutes, 16 secondsNow, you know, there are many, many consequences of computational irreducibility.

46:2046 minutes, 20 secondsIf you start saying, you know, I'm going to build an AI and I want it only to think good thoughts and do good things.

46:2646 minutes, 26 secondsWell, the problem is, as soon as you build an AI that is actually making sort of deep use of computation, it's going to have computational irreducibility.

46:3546 minutes, 35 secondsAnd it's going to have this feature that it can always surprise us.

46:3846 minutes, 38 secondsIt can always do things that are the result where you can tell what it does by just following through the steps and seeing what it does.

46:4546 minutes, 45 secondsBut you can never say, I know you're never going to do the wrong thing or whatever else. It's a trade-off, actually.

46:5146 minutes, 51 secondsIt's something that I think will be a feature of sort of societal decisions is do you go for computational reducibility or you go for computational irreducibility?

47:0247 minutes, 2 secondsWe've had the experience sort of after the Industrial Revolution of having machines where we can kind of understand how they work.

47:0947 minutes, 9 secondsThey've got gears and levers and things like this. Before the Industrial Revolution, lots of things we used, we didn't understand. You know, you ride a horse.

47:1747 minutes, 17 secondsThe horse says, we know what we can do with the horse. don't know how the horse works inside.

47:2247 minutes, 22 secondsAnd then, you know, post Industrial Revolution, we did have sort of a way of understanding how our machines work.

47:2747 minutes, 27 secondsBut that's something as we get into this sort of domain of computational machines, that's no longer the case.

47:3547 minutes, 35 secondsAnd we can either say we insist on knowing how the machine works inside. It's got to be computationally reducible in its behavior.

47:4147 minutes, 41 secondsIf it's computationally reducible, it will be very limited in what it can do.

47:4547 minutes, 45 secondsIf we say, no, it can be computationally irreducible, then we can make use of its computational capabilities to the fullest extent.

47:5347 minutes, 53 secondsBut then it has the problem that in principle, it can have, things can happen which will be surprises to us and which we can't foresee in advance.

48:0148 minutes, 1 secondAnd sort of one can ask, what does that mean for sort of us coexisting with the AIs and so on? I I mean, I think already we're in a situation where in addition to human civilization, there's a civilization of the AIs.

Chapter 13: AI as a civilization we're learning to coexist with

48:1248 minutes, 12 secondsAnd the question is, what is it like to have a world in which there's this alien civilization right in front of us doing all these things?

48:2148 minutes, 21 secondsWell, actually, we have a very common experience of that, which is nature.

48:2648 minutes, 26 secondsNature is, we can think of it also like a sort of alien civilization that's doing things, that's computing all these different kinds of things.

48:3348 minutes, 33 secondsWe've learned to coexist with nature.

48:3548 minutes, 35 secondsYou know, we build houses that prevent it, you know, problems when it rains and things like this.

48:4048 minutes, 40 secondsI mean, that's the feature of kind of this sort of computationally irreducible technology, different from sort of

48:4748 minutes, 47 secondsof the traditional engineering tradition of saying, we build only things where we can foresee what they're going to do.

48:5448 minutes, 54 secondsAs soon as we start really making use of the computational universe, we break away from, we're building only things where we can foresee what they will do.

49:0249 minutes, 2 secondsOne of the things that comes about by thinking about the ruliad and sort of everything that can be, everything that is, and so on is,

49:1249 minutes, 12 secondsone might have thought that science, the universe, is sort of a cold, inhuman kind of place.

49:2049 minutes, 20 secondsAnd I think what has come out from the science I've done is that an awful lot of science reflects back on us humans in very important ways.

49:2949 minutes, 29 secondsIn other words, there's in a sense nothing to say if there isn't a human somewhere in the middle.

49:3449 minutes, 34 secondsIn other words, without an observer, without an observer with definite characteristics, everything is kind of, there's only kind of very uniform things to say.

49:4349 minutes, 43 secondsSo when it comes to kind of when we talk about sort of AI and is there something sort of different and special about us humans, the answer is yes.

49:5249 minutes, 52 secondsThe whole bundle of things that make up the human condition is unique. It is that whole bundle of things.

49:5849 minutes, 58 secondsAnd the AIs that don't have mortality or don't have certain kinds of sensory experiences or whatever, they are different in those ways from us humans.

50:0850 minutes, 8 secondsNow, interesting question for us as humans. You might say at some point, enough is enough.

50:1550 minutes, 15 secondsYou know, we in our technology, what is technology?

50:1950 minutes, 19 secondsTechnology is kind of taking what exists in the world and applying it for human purposes.

50:2450 minutes, 24 secondsFinding that, you know, that magnetic material, we can use that to, you know, snap things together or make a compass.

50:3150 minutes, 31 secondsWe can use those liquid crystals to make a display.

50:3450 minutes, 34 secondsWe're taking things from the natural world and we're kind of applying them for human purposes. And this idea of computational irreducibility.

50:4250 minutes, 42 secondsand so on, tells us we're always going to be able to find more things that we can apply from the natural world. And the question is, well, at what point is sort of, are we done?

50:5250 minutes, 52 secondsAt what point is, you know, between our computational systems and our AIs and our robotics and all that kind of thing, at what point have we got everything that we need to have?

51:0451 minutes, 4 secondsI don't think that's the nature of us as biological organisms.

51:0951 minutes, 9 secondsI think we are, to some extent, we have the vestiges of natural selection, of the sort of struggle for life over the last three billion years.

51:1951 minutes, 19 secondsWe are continually kind of seeking the new. That has been the that's been the experience to this point.

51:2751 minutes, 27 secondsSo, you know, the idea that kind of we're done now is unlikely to be what will happen. But we don't need to be done.

51:3451 minutes, 34 secondsThere's an infinite amount of things that can be discovered. It's not like every invention that can be made has already happened.

51:4151 minutes, 41 secondsAnd then what becomes important is the sort of the human choice of which possibility to pursue.

51:4751 minutes, 47 secondsSo within the computational universe, there's sort of lots of sort of infinite collection of things that can happen that one can study.

51:5551 minutes, 55 secondsThe thing that is the role of us humans is to decide which particular things should we do and look at and so on. What should we choose to find interesting?

52:0452 minutes, 4 secondsI mean, it's something in ruliology as one studies sort of possible simple programs.

52:0952 minutes, 9 secondsIt's a thing that is sort of a routine experience. You can pick a random simple program. You look at it, you say, that's really cool what it does.

52:1852 minutes, 18 secondsIt makes a nice picture. I don't know what its significance is.

52:2152 minutes, 21 secondsIt's not connected to things that are part of what my human experience, our collective human experience has led us to think about.

52:3152 minutes, 31 secondsIt's just like in human language, you know, we have 50,000 words in typical languages. We've given words to certain kinds of things that we care about.

52:3952 minutes, 39 secondsThere's a lot of other things that we can imagine in the abstract world, imagine in the ruliad etc. that we have not yet humanized.

52:4852 minutes, 48 secondsWe've not yet thought that those things are things that is worth us discussing, that we should include as part of kind of the way we think about things in our lives.

52:5752 minutes, 57 secondsAnd it's been a very exciting journey the last few years, kind of exploring those things.

53:0353 minutes, 3 secondsAnd I suppose from a philosophical point of view, one of the things that has most surprised me is that it looks as if, in some sense, we can derive the laws of physics.

53:1353 minutes, 13 secondsI had always assumed that the laws of physics that we get in our universe are kind of just things that get wheeled into the universe, that our universe happens to have the laws it does.

53:2253 minutes, 22 secondsThere's this inevitability for observers like us that we must perceive the laws of physics as they are.

53:2953 minutes, 29 secondsIf we were observers not like us, we might perceive different laws of physics.

53:3353 minutes, 33 secondsBut actually what seems to be the case is that for entities like us, for observers like us, it is inevitable and derivable that the laws of must have the form that they do.



Sync to video time