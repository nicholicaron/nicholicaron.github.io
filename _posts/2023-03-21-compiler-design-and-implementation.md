15-411
Compiler Design
15-411
Carnegie Mellon Course that I will be Auditing

It's been a long time since my last post. In addition to continuing my Math undergrad, I've been working in IT -- first as a Helpdesk Technician, and more recently as a System Administrator. It has been hard to find the time for my personal projects. I would spend a spare hour here and there but I was missing out on the economies of scale that accompany deep, uninterrupted focus. Several of my peers' career goals were to become System Administrators, and I had reached that point without much effort. While I could have continued down the IT path and lived a very comfortable life, I felt that I was still settling. Two quotes really sparked something within me to take a step back and go all in on my goals:

    "Will inertia be your guide or will you follow your passions?"
    - Jeff Bezos

    "The important thing is this: to be able at any moment to sacrifice what we are for what we could become"
    - Charles Du Bos

I felt that I had to give up good to get to great. I also realized that now is the time to take risks -- while I am still nimble, before I have the major responsibilities that come with adulting. With that being said, let's get this show on the road...

** CAN WE INSERT LINKS TO THE OTHER COMPILER POSTS?**

Why learn about compilers?

    "If you don't know how compilers work, then you don't know how computers work. If you're not 100% sure whether you know how compilers work, then you don't know how they work."
    - Steve Yegge

Compilers (and interpreters) are incredibly complex systems that us programmers use on a day-to-day basis. They are what allows us to abstract away the details of the hardware that runs the code we write -- reducing our mental load. This affords us great leverage, and provides automated double-checking of our code to help ensure correctness. By better understanding the tools of the trade, we become more efficient artisans.

    "For the programmer, compiler construction reveals the cost of the abstractions that programs use... A programmer cannot effectively tune an application's performance without knowing the costs of its individual parts."
    - Cooper & Torczon (Engineering a Compiler)

Additionally, with the end of Dennard Scaling and Moore's Law, we can no longer rely on regular improvements in clock speeds to improve our programs performance for us. The future of speedups will likely have to come from improvements in architecture and parallelism -- through improvements in optimization and code generation via specialized hardware/software codesign.

Personally, I tend to admire Systems programmers (and more generally, people who choose the more difficult paths in life). Since a common consequence of admiration is emulation, this -- mixed with a tinge of masochism and a proclivity for patterns -- has led me to develop a penchant for tinkering with low-level systems.

I also enjoy learning human languages; in high school I took 5 Spanish classes, in addition to Arabic and American Sign Language. Also, my dad's side of the family are Francophones from Quebec, so that's next on my list to learn. Languages in general are all about expressing intent (semantics) via an agreed upon form (syntax) and abstractions. One of the primary differences between spoken languages and programming languages is the level of ambiguity.

Compilers are the subject of numerous garguantuan books, therefore my goal with this blog post is to only provide a rough approximation. The information provided here is by no means exhaustive.

    "You’re here, ultimately, because of control. Your entire life is entombed in an elaborate, entirely abstract labyrinth of machines that define so many aspects of every moment of your waking life. You’re here to better understand the prison you find yourself within. You’re here to sympathize with the machine."
    - Tyler Neely

https://youtu.be/6WxJECOFg8w
The real reason to learn about compilers is so we can utilize the infamous Double Compile whenever we find ourselves in desperate times.

Why use Rust? 

Well, if the question is a general one, see this previous article. If you mean why use Rust for writing compilers, then see the introduction section to this online book. It is extremely important for the literal thing that creates your binaries that you run on your machine to, itself, be as robust as possible for obvious reasons. To quote David Wheeler's paper Countering Trusting Trust through Diverse Double-Compiling: "compilers can be subverted to insert malicious Trojan horses into critical software, including themselves."

If there are vulnerabilities in your compiler, it is difficult to trust the code it generates. Compilers are huge, complicated pieces of software (often on the order of 100K-1M LOC). I am by no means an elder greybeard GNU wizard who dreams in C, and therefore I would definitely not trust myself to make this thing in C/C++. Since this is not a production compiler, security is not of paramount importance, but we may discuss some security-related topics.
The Anatomy of a Compiler
Note: This is a generalization. In practice, not all compilers share the same structure

    "A successful compiler executes an unimaginable number of times... Thus, compiler writers must pay attention to compile time costs, such as the asymptotic complexity of algorithms and the space used by data structures."
    - Cooper & Torczon (Engineering a Compiler)

Written Assignments
Assignment 1: Backend
Assignment 2: Frontend
Assignment 3: Middle
Assignment 4: Semantics
Continuing with this project – What's next? 

    Let's use programmable logic and FPGA's to create a RISC-V board to run our compiled binaries on!
    Let's also, beef up our compiler with Cornell's CS 6120: Advanced Compilers: The Self-Guided Online Course

Primary References

    Engineering a Compiler
    Crafting Interpreters
    Computer Organization and Design RISC-V Edition
    Hacker's Delight
    https://llsoftsec.github.io/llsoftsecbook/

Background Reading:

    https://steve-yegge.blogspot.com/search?q=compiler (Great, comical read on why we should study compilers at all. Although it is notably dated in some areas – e.g. concerning static typing (Rust FTW))
    https://omscs.gatech.edu/cs-8803-o08-compilers-theory-and-practice-course-videos
    https://belkadan.com/blog/2016/05/So-You-Want-To-Be-A-Compiler-Wizard/
    https://www.inapps.net/rust-creator-graydon-hoare-recounts-the-history-of-compilers-inapps-2022/
    https://thume.ca/2019/04/18/writing-a-compiler-in-rust/
    https://llvm.org/docs/tutorial/
    https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl01.html
    https://llvm.org/devmtg/2019-04/talks.html
    https://www.cs.uaf.edu/users/chappell/public_html/class/2018_spr/cs331/docs/types_primer.html
    https://borretti.me/article/lessons-writing-compiler

Memes for the road