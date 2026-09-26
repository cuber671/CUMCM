%
% gmcmthesis.cls —— “华为杯”第二十三届中国研究生数学建模竞赛论文模板
% 基座：latexstudio/GMCMthesis v2.4 (2023/09/19, andy123t)
% 本版：v2.5-23rd，按官方《附件3 论文模板(.doc)》《附件2 论文格式规范》逐项实测适配
%   - 页面：A4，上 30mm / 下 20mm / 左右 22.5mm（附件3 sectPr 实测值）
%   - 行距：baselinestretch=1.30（附件3 docGrid 312tw ÷ 小四 12pt，即“单倍行距”）
%   - 封面：4 个官方 logo（36.1×23.0 / 27.6×26.6 / 28.0×28.0 / 24.1×22.8 mm，基线对齐居中）
%           “中国研究生创新实践系列大赛” 小二华文新魏加粗
%           主标题两行 二号华文新魏加粗
%           学校/参赛队号/队员姓名 小二加粗 + 下划线
%   - 摘要页：题目栏 隶书小二标签 + 三号黑体题目（规范：题目用三号黑体）
%           “摘 要：”“关键词：” 隶书小二
%   - 页码：封面页脚居中显示"0"（官方模板原样），摘要页起从 1 连续编号、页脚居中、无页眉
%   - 正文：小四宋体；一级标题四号黑体居中；图表编号按章；参考文献 gmcm.bst
%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{gmcmthesis}[2026/09/23 v2.5-23rd GMCM thesis class]

%% Options
\newif\if@gmcm@bwprint\@gmcm@bwprintfalse
\newif\if@gmcm@preface\@gmcm@prefacetrue
\newif\if@gmcm@coverzero\@gmcm@coverzerotrue   % 封面页脚是否显示页码 0（官方模板默认显示）
\DeclareOption{colorprint}{\@gmcm@bwprintfalse}
\DeclareOption{bwprint}{\@gmcm@bwprinttrue}
\DeclareOption{withoutpreface}{\@gmcm@prefacefalse}
\DeclareOption{nocoverzero}{\@gmcm@coverzerofalse} % 若提交系统不希望封面出现 0，用此选项
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexart}}
\ExecuteOptions{colorprint}
\ProcessOptions\relax

\LoadClass[a4paper,zihao=-4]{ctexart}

%% Packages
\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{gmcmthesis}{You must use the `xelatex' driver}{Just choose `xelatex'.}
\fi
\RequirePackage{geometry}
\RequirePackage{amsmath,amsfonts,amssymb,bm}
\RequirePackage{color,xcolor}
\RequirePackage{graphicx}
\RequirePackage{tabularx,array,makecell,longtable,booktabs,multirow,bigstrut,bigdelim}
\RequirePackage{cprotect}
\RequirePackage{listings}
\RequirePackage{indentfirst}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{ulem}
\RequirePackage{calc}
\RequirePackage{titletoc,url}
\RequirePackage[numbers]{natbib}
\setlength{\bibsep}{0pt plus 0.3ex}
\RequirePackage{etoolbox}
\AtBeginEnvironment{thebibliography}{%
    \phantomsection
    \addcontentsline{toc}{section}{\refname}}
\RequirePackage{hyperref}

%% 代码高亮颜色
\definecolor{mKeyword}{RGB}{0,0,255}
\definecolor{mString}{RGB}{160,32,240}
\definecolor{mComment}{RGB}{34,139,34}
\definecolor{pKeyword}{RGB}{228,0,128}
\definecolor{pString}{RGB}{148,0,209}
\definecolor{pComment}{RGB}{117,113,94}

\lstnewenvironment{Matlab}[1]{%
\lstset{language=matlab,
  xleftmargin=20pt, xrightmargin=20pt,
  frame=tlbr, framesep=4pt, framerule=0.5pt,
  basicstyle=\small\ttfamily,
  keywordstyle={\color{mKeyword}},
  stringstyle={\color{mString}},
  commentstyle={\color{mComment}},
  backgroundcolor=\color{gray!5},
  rulecolor=\color{gray!50},
  title=#1,
  keywords={break,case,catch,classdef,continue,else,elseif,end,for,
  function,global,if,otherwise,parfor,persistent,return,spmd,switch,try,while},
  showspaces=false, showstringspaces=false, showtabs=false,
  tabsize=4, captionpos=t, breaklines=true,
}}{}

\lstnewenvironment{Python}[1]{%
\lstset{language=python,
  xleftmargin=20pt, xrightmargin=20pt,
  frame=single, framesep=4pt, framerule=0.5pt,
  basicstyle=\small\ttfamily,
  keywordstyle=\color{pKeyword},
  stringstyle=\color{pString},
  commentstyle=\color{pComment},
  backgroundcolor=\color{gray!5},
  rulecolor=\color{gray!50},
  showspaces=false, showstringspaces=false, showtabs=false,
  tabsize=4, breaklines=true,
}}{}

\lstset{
  frame=tb, xleftmargin=25pt, aboveskip=3mm, belowskip=3mm,
  showstringspaces=false, columns=flexible, framerule=1pt,
  rulecolor=\color{gray!35}, backgroundcolor=\color{gray!5},
  basicstyle={\small\ttfamily}, numbers=none,
  numberstyle=\tiny\color{gray}, keywordstyle=\color{blue},
  commentstyle=\color{green!50!black}, stringstyle=\color{purple},
  breaklines=true, breakatwhitespace=true, tabsize=3,
}

%% 页面布局（附件3 实测：上 30mm、下 18.5mm+页脚区、左右 22.5mm；页码位于页脚中部）
\geometry{a4paper,top=30.0mm,bottom=20.0mm,left=22.5mm,right=22.5mm,
  headheight=0mm,headsep=0mm,footskip=2.5mm}
\renewcommand*{\baselinestretch}{1.30} % 单倍行距（docGrid 312tw/12pt）

% 表格行距与正文一致
\let\gmcm@oldtabular\tabular
\let\gmcm@endoldtabular\endtabular
\renewenvironment{tabular}%
  {\bgroup\renewcommand{\arraystretch}{1.38}\gmcm@oldtabular}%
  {\gmcm@endoldtabular\egroup}
\setlength\parindent{2em}

%% 字体（西文）
\newif\ifmcm@mac
\newif\ifmcm@windows
\newif\ifmcm@fandol
\@ifclasslater{ctexart}{2015/05/06}{%
  \newcommand\mcm@strifeq[4]{%
  \edef\ustc@tmp@a{#1}\edef\ustc@tmp@b{#2}%
  \ifx\ustc@tmp@a\ustc@tmp@b #3\else #4\fi}
  \newcommand\mcm@fontset{\csname g__ctex_fontset_tl\endcsname}
  \mcm@strifeq{\mcm@fontset}{mac}{\mcm@mactrue}{}
  \mcm@strifeq{\mcm@fontset}{windows}{\mcm@windowstrue}{}
  \mcm@strifeq{\mcm@fontset}{fandol}{\mcm@fandoltrue}{}
}{\mcm@windowstrue}

\ifmcm@fandol
  \setmainfont[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
    ItalicFont=*-italic,BoldItalicFont=*-bolditalic]{texgyretermes}
  \setsansfont[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
    ItalicFont=*-italic,BoldItalicFont=*-bolditalic]{texgyreheros}
  \setmonofont[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
    ItalicFont=*-italic,BoldItalicFont=*-bolditalic,Scale=MatchLowercase]{texgyrecursor}
\else
  \setmainfont{Times New Roman}
  \setsansfont{Arial}
  \ifmcm@mac
    \setmonofont[Scale=MatchLowercase]{Menlo}
  \else
    \setmonofont[Scale=MatchLowercase]{Courier New}
  \fi
\fi

%% 字体（中文）：宋体/黑体/楷体 + 模板专用 华文新魏(标题) 隶书(题目/摘要/关键词标签)
\ifmcm@mac
  \setCJKmainfont[UprightFont=* Light,BoldFont=* Bold,
    ItalicFont=Kaiti SC,BoldItalicFont=Kaiti SC Bold]{Songti SC}
  \setCJKsansfont{Heiti SC}
  \setCJKfamilyfont{zhsong}[UprightFont=* Light,BoldFont=* Bold]{Songti SC}
  \setCJKfamilyfont{zhhei}{Heiti SC}
  \setCJKfamilyfont{zhkai}{Kaiti SC}
  \setCJKfamilyfont{zhxw}[AutoFakeBold=2.15]{STXinwei}      % 华文新魏
  \setCJKfamilyfont{zhli}{LiSu}                              % 隶书
\else
  \ifmcm@windows
    \IfFileExists{C:/bootfont.bin}{%
      \setCJKmainfont[AutoFakeBold=2.15,ItalicFont=KaiTi_GB2312]{SimSun}%
      \setCJKfamilyfont{zhkai}[AutoFakeBold=2.15]{KaiTi_GB2312}%
    }{%
      \setCJKmainfont[AutoFakeBold=2.15,ItalicFont=KaiTi]{SimSun}%
      \setCJKfamilyfont{zhkai}[AutoFakeBold=2.15]{KaiTi}%
    }
    \setCJKsansfont[AutoFakeBold=2.15]{SimHei}
    \setCJKfamilyfont{zhsong}[AutoFakeBold=2.15]{SimSun}
    \setCJKfamilyfont{zhhei}[AutoFakeBold=2.15]{SimHei}
    \setCJKfamilyfont{zhxw}[AutoFakeBold=2.15]{STXinwei}     % 华文新魏
    \setCJKfamilyfont{zhli}{LiSu}                             % 隶书
  \else
    \ifmcm@fandol % Linux/Overleaf：无华文新魏/隶书，以楷体近似替代
      \setCJKfamilyfont{zhkai}{FandolKai-Regular.otf}
      \setCJKfamilyfont{zhxw}[AutoFakeBold=2.15]{FandolKai-Regular.otf}
      \setCJKfamilyfont{zhli}[AutoFakeBold=2.15]{FandolKai-Regular.otf}
      \setCJKfamilyfont{zhhei}[BoldFont=FandolHei-Bold.otf]{FandolHei-Regular.otf}% 摘要页题目加粗（2026-09-26）
      \ClassInfo{gmcmthesis}{fandol 字体集下以楷体近似替代华文新魏/隶书；%
        追求像素级一致请在 Windows/macOS 字体环境下编译}
    \fi
  \fi
\fi
\def\lishu{\CJKfamily{zhli}}
\def\xwxwei{\CJKfamily{zhxw}}

%% 数学定理环境
\newtheorem{definition}{定义}
\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{assumption}{假设}
\newtheorem{example}{例}
\newtheorem{proof}{证明}
\newtheorem{solution}{解}

%% 浮动体与图表
\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.jpeg,.png}
\graphicspath{{figures/}{figure/}{pictures/}{picture/}{pic/}{pics/}{image/}{images/}}
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{format=hang,labelsep=quad,font={song,minusfour},position=bottom}
\captionsetup[table]{format=hang,labelsep=quad,font={song,minusfour},position=top}

%% 列表
\setlist{topsep=0.3em,partopsep=0pt,itemsep=0ex plus 0.1ex,parsep=0pt,
  leftmargin=1.5em,rightmargin=0em,labelsep=0.5em,labelwidth=2em}

%% 超链接
\hypersetup{hyperfootnotes=true,hyperindex=true,colorlinks=true,
  bookmarksnumbered=true,bookmarksopen=true,bookmarksopenlevel=0,
  allcolors=black,breaklinks=true}

%% tokens
\newcommand\gmcm@tokens@keywords{}
\newcommand*\gmcm@tokens@baominghao{}
\newcommand*\gmcm@tokens@schoolname{}
\newcommand*\gmcm@tokens@membera{}
\newcommand*\gmcm@tokens@memberb{}
\newcommand*\gmcm@tokens@memberc{}
\newcommand*\baominghao[1]{\renewcommand{\gmcm@tokens@baominghao}{#1}}
\newcommand*\schoolname[1]{\renewcommand{\gmcm@tokens@schoolname}{#1}}
\newcommand*\membera[1]{\renewcommand{\gmcm@tokens@membera}{#1}}
\newcommand*\memberb[1]{\renewcommand{\gmcm@tokens@memberb}{#1}}
\newcommand*\memberc[1]{\renewcommand{\gmcm@tokens@memberc}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 封面（附件3 第 1 页，逐元素按实测坐标复刻）
\def\gmcm@coverpage{%
  \clearpage
  \pdfbookmark[1]{封~面}{cover}%
  \setcounter{page}{0}%
  \if@gmcm@coverzero\thispagestyle{plain}\else\thispagestyle{empty}\fi
  \begingroup
    \centering
    % logo 行：官方 4 logo 原始尺寸（mm），inline 基线对齐
    \vspace*{-3.0mm}%
    \includegraphics[width=36.1mm,height=23.0mm]{logo1.png}%
    \includegraphics[width=27.6mm,height=26.6mm]{logo2.png}%
    \includegraphics[width=28.0mm,height=28.0mm]{logo3.jpeg}%
    \includegraphics[width=24.1mm,height=22.8mm]{logo4.png}\par
    \vskip 10.2mm
    {\zihao{-2}\xwxwei\bfseries 中国研究生创新实践系列大赛\par}
    \vskip 2.0mm
    {\zihao{2}\xwxwei\bfseries “华为杯”第二十三届中国研究生\par}
    \vskip 4.0mm
    {\zihao{2}\xwxwei\bfseries 数学建模竞赛\par}
  \endgroup
  \vskip 24.5mm
  % 信息表：学校 / 参赛队号 / 队员姓名，小二加粗，值列下划线
  \renewcommand\arraystretch{1.4}%
  \noindent
  \begin{tabularx}{\textwidth}{@{}p{30mm}X@{}}
    \zihao{-2}\bfseries 学\hspace{1.8em}校 &
      \hfill\zihao{-2}\bfseries \gmcm@tokens@schoolname\hfill\null \\[2mm]
    \Xhline{1.2pt}
    \noalign{\vskip 9mm}
    \zihao{-2}\bfseries 参赛队号 &
      \hfill\zihao{-2}\bfseries \gmcm@tokens@baominghao\hfill\null \\[2mm]
    \Xhline{1.2pt}
    \noalign{\vskip 12mm}
    \zihao{-2}\bfseries 队员姓名 &
      \zihao{-2}\bfseries 1.~\gmcm@tokens@membera\hfill\null \\[2.8mm]
    & \zihao{-2}\bfseries 2.~\gmcm@tokens@memberb\hfill\null \\[2.8mm]
    & \zihao{-2}\bfseries 3.~\gmcm@tokens@memberc\hfill\null \\[2mm]
    \Xhline{1.2pt}
  \end{tabularx}\par
  \vskip 4mm
  \clearpage
}

%% 摘要页页首（附件3 第 2 页：无 logo 的标题块 + 题目栏）
\def\gmcm@titlepagehead{%
  \setcounter{page}{1}%
  \begingroup
    \centering
    \vspace*{-4.4mm}%
    {\zihao{-2}\xwxwei\bfseries 中国研究生创新实践系列大赛\par}
    \vskip 1.9mm
    {\zihao{2}\xwxwei\bfseries “华为杯”第二十三届中国研究生\par}
    \vskip 4.0mm
    {\zihao{2}\xwxwei\bfseries 数学建模竞赛\par}
  \endgroup
  \vskip 12.5mm
  \begin{center}
  \begin{tabularx}{.96\textwidth}{@{}lX@{}}
    {\zihao{-2}\lishu 题\quad 目：} &
      \hfill\zihao{3}\heiti\bfseries\@title\hfill\null \\
    \cline{2-2}
  \end{tabularx}
  \end{center}
  \vskip 13.7mm
}

%% 摘要环境：“摘 要：” 隶书小二居中（官方模板原式）
\renewenvironment{abstract}{%
  \if@twocolumn
    \section*{\gmcm@cap@abstractname}%
  \else
    \begin{center}%
    {\zihao{-2}\lishu\bfseries\gmcm@cap@abstractname\vspace{\z@}}% 摘 要：标签加粗（2026-09-26 用户要求）
    \end{center}%
    \quotation
  \fi}
  {\if@twocolumn\else\endquotation\newpage\fi}
\renewenvironment{quotation}
  {\list{}{\listparindent 2em
   \itemindent \listparindent
   \rightmargin\z@ \leftmargin\z@
   \parsep \z@ \@plus\p@}%
  \item\relax}
  {\endlist}

%% 关键词：隶书小二（附件3 实测 18pt 隶书，v2.4 的小三已按官方模板修正为小二）
\newcommand\keywords[1]{%
  \renewcommand{\gmcm@tokens@keywords}{#1}%
  \par\vskip 2.5ex
  {\noindent\zihao{-2}\lishu\gmcm@cap@keywordsname：}~{\zihao{-4}\songti\gmcm@tokens@keywords}%
  }

\newcommand*{\gmcm@cap@abstractname}{摘\quad 要：}
\newcommand*{\gmcm@cap@keywordsname}{关键词}

%% \maketitle：封面 + 摘要页页首
\renewcommand{\maketitle}{\par
  \begingroup
    \newpage
    \global\@topnum\z@
    \if@gmcm@preface
      \gmcm@coverpage
      \gmcm@titlepagehead
    \fi
  \endgroup
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@author\@empty
  \global\let\@date\@empty
}

%% 章节标题：一级四号黑体居中（规范），二三级左对齐加粗
\setcounter{secnumdepth}{3}
\def\@seccntformat#1{\csname the#1\endcsname\ \ }
\newcommand{\thinskip}{\hskip 0.05em\relax}
\renewcommand\thesubsection{\arabic{section}\thinskip.\thinskip\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection\thinskip.\thinskip\arabic{subsubsection}}
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {2.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\bfseries\centering\zihao{4}\heiti}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {1.25ex\@plus -1ex \@minus -.2ex}%
  {1.25ex\@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {1.25ex\@plus -1ex \@minus -.2ex}%
  {1.2ex\@plus .2ex}%
  {\normalfont\normalsize\bfseries}}

%% 图表公式编号按章
\renewcommand{\thetable}{\arabic{section}.\arabic{table}}
\renewcommand{\theequation}{\arabic{section}.\arabic{equation}}
\renewcommand{\thefigure}{\arabic{section}.\arabic{figure}}

%% 目录
\RequirePackage[titles]{tocloft}
\renewcommand{\cftsecdotsep}{4.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\heiti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecaftersnumb}{\hskip.4em}
\newcommand{\maketoc}{%
  \phantomsection
  \pdfbookmark[1]{目~录}{contents}%
  \tableofcontents}

%% 附录
\RequirePackage[title,titletoc]{appendix}
\newcommand*{\gmcm@cap@appendixname}{附录}
\renewcommand*\appendixname{\gmcm@cap@appendixname}
\renewcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\appendixname\@Alph\c@section}}

\endinput
