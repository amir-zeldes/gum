# -*- coding: UTF-8 -*-

import io, os, sys, re
from collections import defaultdict
from argparse import ArgumentParser
from glob import glob

from scipy.stats import zscore


class Token:
    def __init__(self, abs_id, form, xpos, abs_head, deprel, speaker):
        self.id = abs_id
        self.form = form
        self.xpos = xpos
        self.head = abs_head
        self.deprel = deprel
        self.speaker = speaker

    def __repr__(self):
        return self.form + "/"+self.xpos+ "("+str(self.id)+"<-"+self.deprel+"-"+str(self.head)+")"


headers = ["doc","unit1_toks","unit2_toks","unit1_txt","unit2_txt","s1_toks","s2_toks","unit1_sent","unit2_sent","dir","orig_label","label"]

# stop lists, lowercased, plus <*>
stop = {}
stop["eng"] = {"<*>", ",", ".", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
stop["deu"] = set("""<*> . , einem nahm recht dafür beispiel dasselbe warum davon grosses diejenigen wahr habt hier kaum ob gemusst niemanden ab heute weitere kleiner anderem jeder siebente früher hätte durch deshalb derselben neunten etwa jedem zwanzig hast ganze um neunter durchaus wohl ohne also auch rechter zeit sollte lange darunter welcher dich wer siebenter jedoch beide sehr siebtes zugleich wo deren seine sieben außer da wann einmaleins auf zehnten solchem dermassen jede dazu je gedurft siebter seien diese dann gut erster ihnen einander man ganz entweder meinem zusammen erste immer besonders dermaßen rechtes aus bisher damit achtes kleinen gegenüber kommen deswegen wenigstens dieselbe geworden mich gute grosse im gegen dasein sondern musste allem unter müsst dementsprechend gab durften seines achter darauf noch sonst manchen gesagt elf weniger aller acht weiter keinem dank darfst á ihrer kleines zurück mancher hätten wirst wenig demgemäss ging nun ihm mochte rechte seiner derjenigen grosser gleich zuerst mögt ersten waren beim wem meine dieselben musst das so dürft wessen dort rund davor dritten hatte tage dahin eigenes kurz viele solches etwas satt sich wurden drittes nein ebenso jahr solche weniges ihn dir kleine sagt vielleicht weiteren keine habe ausserdem demgemäß fünften achte neunte mochten würde keinen kam demzufolge allerdings jetzt vielem fünfter dahinter wollen dieses sechsten rechten solang hat sechster in guter wegen schlecht würden na macht zwischen tat zu wollte eines besser gerade eigene zehntes mehr mögen müssen trotzdem darin ihr muß zum großer neuen jene tagen bist alles bin eigenen jedermanns zwei damals jahren endlich hatten dessen andere lang vor vielen es über heisst zweiten alle an wart mein siebte seinen indem ihres neun welchen ach dein irgend den gehabt meinen her andern große tag mittel zunächst sein oben dies hinter lieber neben a meiner fünftes zehn unser was gross ihren jemanden ein sie wird allein jahre werdet manchem außerdem anders ausser wieder daneben dritte worden teil war können erstes währenddessen seit offen währenddem darf allen dass geschweige danach gewollt werde eine sechstes achten darüber ende oft zur zehnte jemand zweiter einige muss daraus euch diesen wäre ihrem neue groß richtig infolgedessen diesem doch deiner geht diejenige vierten weil ist konnten solchen ehrlich wollten nachdem morgen siebentes durfte kannst von wurde wenige jener denn manche eben möglich bereits ganzer leicht los großen denselben grossen nur jenen ag könnt gern zwar werden selbst denen und eigener mir ganzes übrigens heißt weit fünfte solcher gekannt welchem mussten derselbe deine uhr allgemeinen bekannt vierter wenn zweite en gehen daß demgegenüber magst gewesen natürlich jemandem bis jenes siebenten der statt unserer wen vom machen dazwischen drin sei nach dabei zweites die weiteres du derjenige hoch seinem aber uns ich gar tel viel einigen nichts niemandem welches darum besten desselben kein machte mit ja mag ganzen viertes tun beiden anderen sah dieser niemand genug vergangene bald einen deinem willst ins schon vierte dem einiges am nie während daher kann seitdem überhaupt dürfen könnte seid sowie daran dritter jedermann einer als neuntes zehnter erst einmal des für dagegen wirklich sechs sollen hin oder vier daselbst gemocht vergangenen sollten manches sind er haben drei ihre wie soll keiner möchte dadurch sechste jeden später nicht siebten eigen kommt sagte gutes fünf gibt unsere einiger welche konnte jenem großes meines leider wollt bei gemacht wir demselben will gekonnt""".split())
stop["eus"] = set("""<*> . , zu da haietan nola berau hi zein zuen handik ditu hemen hura ze hango nor horri zuten hor horrela ere hartan haiei horiei anitz dute berauek baina noiz hari batean batzuetan horietan gu honela dago hara beraiek haiek honetan ez du eta hona zergatik hauetan honek ziren hauei horrek zituen nora izan beroriek bere hauek batzuei beste bera gainera bati eurak zenbait zen batek hau non ni bezala dira hainbeste horiek edo horko hark bat batzuk berori nongo arabera hortik egin horra horretan honi al batzuek nondik hala zer guzti zenbat hori asko gutxi han hemengo zuek hemendik""".split())
stop["fra"] = set("""<*> . , toi-meme exactement mienne outre celui-ci mien antérieures près y dits différentes dix différent quel nouveau nous faisant differente aura desquels vont cinq huitième me surtout autrement au plus précisement également car quoi comment derriere mille pu ce ès restent plusieurs siennes ni hue eu voila semblable pense ha dessous ai autres relative elles-memes ne allaient un celle cent sont vé soixante suffisant n' maintenant pourrait premier sur six toutes aussi semble suis deuxièmement moi-même voici parle antérieure feront memes quatre votre troisième rendre divers lesquelles vas unes spécifiques etc très quelqu'un seulement celle-la les s’ anterieures l' du qu’ eux-mêmes neuvième da suivre jusque dedans toute dix-sept desormais fais autrui envers quatrième t’ egalement il abord suit si uns mon pas different siens étais meme attendu la antérieur aupres m’ ton cependant ah quelque tres procedant suffisante certain chez sent deux entre s' ceci va es quant-à-soi quelques premièrement auxquelles ouias douzième neanmoins sans debout pourrais autre quelles vous-mêmes cela delà toi peut pourquoi specifiques tien celles chacun suivante tiens certes elles quelconque retour quinze font peux etais qu' celui-la lesquels devant parfois spécifique n’ soi-même assez vu d’ nul troisièmement eux basee moindres ont precisement reste avais ma duquel nombreux mêmes ho diverse quelle ouverts à où tend tous ouste prealable merci relativement cinquième certaine c’ des jusqu revoici seuls desquelles hi vôtres pouvait dix-neuf avec ô facon le chacune encore par etaient tout durant différente miens effet j’ lui te pendant suffit devra aux malgre seules déja onze ta depuis semblaient onzième étant peu rend puisque hé hou quoique on touchant nos combien â gens ceux-ci pour bas lequel maint auraient celles-ci i concernant derrière seule nôtre houp partant était eh cinquantième moins etant suivantes ces suivants lui-même laisser ouverte hors quant dire as ci auxquels sera dix-huit enfin une ou differentes dans ils quatorze dehors sept miennes d' devers avaient directe l’ parmi celui-là malgré aie mais specifique trois ça dit dixième seraient deja qui tente etait via chaque afin votres quarante vais celle-ci je son telle faisaient façon quatrièmement même souvent té voilà na se tenant huit revoilà quiconque elles-mêmes possible dessus seize selon sinon soi juste là quatre-vingt celui ceux sauf hem permet moi-meme toi-même o cette est cinquantaine certains vôtre doivent c' celles-la apres de soi-meme elle restant déjà ayant peuvent tu ouvert aurait comme sait mes doit vos cinquante fait nombreuses tels treize environ douze néanmoins lui-meme tel tes cet trente celles-là tiennes auront leur laquelle bat t' differents telles tant certaines tenir après parler vous longtemps pres personne seront quels stop deuxième puis sous sienne nôtres ceux-là notamment sa directement lors dès avant importe semblent notre que elle-meme plutôt hep celle-là lès donc serait dejà plutot soit avons désormais sien or proche hui alors suivant ainsi vers moi elle-même auquel lorsque allons première septième dont être excepté ses tellement diverses avait a compris dite possibles parlent étaient hormis nous-mêmes différents etre parce revoila préalable quand leurs sixième tienne anterieure j' anterieur et seul avoir toujours en vingt ait m'""".split())
stop["nld"] = set("""<*> . , voordien doch omlaag erdoor afgelopen meesten deze minder doorgaand pp tenzij dan hier dat 't die werden mochten precies waar inmiddels dien eerst wezen hadden gelijk alles der mezelf zekere alle zowat jullie sommige geleden gedurende hen maar aldus gewoonweg op ooit daarna bijna nog hun enige moest tamelijk zij nabij nu over wij vooruit zulke zullen elk allen vooraf min van idd weer zeer verder vooral wiens het eerder is allebei hebt weldra want uw later doorgaans wanneer me effe enkel zulks welken betere of er moeten gegeven jouw wier voorheen hele overigens spoedig zoals opnieuw mijzelf tegen ons voorbij heeft kunt onder jij boven anders en geworden ongeveer eigen alhoewel veel vroeg wilde toen zelfs tijdens naar ze zelf bij al kon eveneens zelfde hoewel beter pas even omdat geen lang zijne nogal ikke hoe sindsdien je gehad zodra mijn vanaf aangezien als in enkele voordat thans verre een bovendien ander kan dezen geheel omtrent misschien niet liever mag gemogen volgens dezelfde men bijvoorbeeld zonder eersten geweest weinige iets beiden zijnde nooit ondertussen vanwege achterna indien ik moesten gewoon steeds wel binnenin zijn tot liet uwe ter onszelf inzake toe vrij werd sinds opzij publ achter mogelijk zou moet u waarom hierboven weinig jou ge daarom terwijl mocht jijzelf daarin waren zoveel zich eens onze ikzelf worden ff beide omhoog hare weg eerste daar welk behalve meer hem uit buiten ook ‘t inz bent wie dit den zouden toch zei hierbeneden uitgezonderd bovenstaand wat om daarop zichzelf was bepaald zeker voorop hij aan bovenal des uwen rondom aangaangde had tussen voor prof ieder jouwe konden alleen na ja omstreeks door daarheen enz haar vooralsnog anderen doen sedert de beneden klaar vandaan vanuit vervolgens anderzijds opdat zal doet zo’n voorts voort evenwel gekund geven krachtens jezelf hebben mede niets kunnen altijd iemand reeds met zo ben ten etc te wegens nr binnen andere vgl zulk slechts hierin rond af betreffende wil juist welke net wordt heb omver veeleer echter nadat elke mogen daarnet dikwijls gij we gauw mij zult dus vaak totdat""".split())
stop["por"] = set("""<*> . , último uns primeira tanta lhe toda tu nossa nunca ambas nos tive vos boa tivestes treze próprio aquilo tivemos sois vindo sim diante dezoito mil estivemos naquela faço puderam tais esses aí cuja certeza aquelas vinte ademais mas baixo teve falta contudo nenhuma posição estava adeus quatro grandes todas de exemplo fará dizem vai dar em pouca quer iniciar pelo oitavo podem fazeis oito somos tem porquê vez dessa grupo poder menos com ela pois não ambos disso favor ao essa sei dezanove naquele está lá os nessa área vêm esse essas já questão minha bastante nova logo através lugar sete mesmo muitos tendes mal tente foi zero ou daquela for nuns dos tiveram comprida estiveste um são eu me desse estive talvez nossas irá usar obrigada o duas seu novo conhecido próxima novas todo quieta cento às segunda na tudo menor fomos fazemos sexto ali dezassete ter fui partir meu debaixo três certamente cedo seis dá cá atrás dentro minhas inicio tipo bem quieto comprido tentaram des fim no também fazia sétima querem porquanto demais você vão este terceiro daquele teus ontem sobre meses temos nem longe pouco tentei coisa número seria ainda dizer sétimo saber e te tens dezasseis põem vossa tiveste nesta fez põe qual se quem nossos mês onze valor relação é quarta quê tua grande forma ir vem sua estão só porém tal outras aos quais nada até pegar estará dão qualquer embora todos tão pontos vais as nesse fazes lado por oitava das conselho agora neste a uma quando somente estás pelas mais vossas quinta além entre aqui outros deste aquele devem ora ele local podia usa sabe apenas meus tempo momento numa muito depois pelos estado eles geral pode sistema ponto sem fora à tuas ver próximo isto outra povo portanto custa cujo umas nós algo nas seus que final meio perto então sempre aqueles do veja conhecida cinco como diz parte obrigado após cada novos és algumas foste bom sou aquela maior contra possível da foram estivestes direita assim estar nível acerca teu doze cima caminho deve estas inclusive apoio poderá faz vezes pôde parece vocês maioria dez tentar eventual tarde primeiro vós desta vens vossos catorze estou onde sexta antes apoia têm apontar fostes para maiorias elas vinda estes dois ser deverá alguns fazem quero tenho quinze porque máximo quanto sob esta nove posso tanto isso pela quinto breve fazer possivelmente números quarto era corrente suas ligado nosso estiveram segundo esteve num desde vários vosso enquanto terceira""".split())
stop["rus"] = set("""<*> . , своем ним одной мой одною этим тот мне при во ей нашему своё моги том наших эту ко нашими своих одними можем себе все своему ты которым вам кому тем саму моей чему свое мы нашего ней наса будет едят ешь моём или вами нас кого кто меня моими ест такие её самому таких сам одному которою будучи ту всей а могли моя своём тобою своею их по сама мою емъ могло собой к нашем одну свою можешь нашу ел всеми вы ею которыми самого мои мое свои же могите будешь тебе томах самом нашей наши собою ем наш неё этою нею нами об который этой моим моем на нём всею мною своими такое которых как кем ними котором будут того которое тех этими такою которому которые моего свой я тою они вся ком нашею своя мог тобой быть своей эта ими всея таким такими такой всём до такому из нем да всю были за мочь одним был нам самим комья могут тебя могу это только этом им таком бы что одна ему чтобы у такого оно чём есть всем чем ела буду в так такую одних ещё самих этих весь мной могла них его когда нет которая чего всему но всего нее одни не этот будь те эти самими себя всех этого оне с имъ теми моё которого было и можете если одном этому для моих та вот которой всё своего будьте едим своим нему один одного она то сами наша тому уже будем него нашим наше была само такая той от вас ее которую моему моею одно о еще будете он может""".split())
stop["spa"] = set("""<*> . , hoy atras sabeis ése una mis trabajo intentas dado mediante mal varios según todas fin acuerdo anterior manera ellas él trata nosotros demás si nuevo hacia sabemos unas nosotras debe mencionó sino actualmente ahora temprano igual momento otro hay demasiado va pero lejos lado para al propias quiénes pasada después encuentra despues hago todavía sea tengo ésa consigues dicho esto sólo breve ningún estado ni consigo ese días existen tendrán muy cerca eres solo consiguen ultimo ésta llegó nuestro tú tras otras vuestro ciertos todavia fui hacer pudo realizado cuándo mí ha última antaño varias vuestros ahí explicó hecho os deben podría segun míos ustedes considera cuanto ninguna aquéllos indicó ninguno mejor deprisa cuatro últimos informo puedo tercera fue enseguida el ademas tiempo entonces mio esos teneis siguiente alrededor cuantos quizá así dias poner aquí cuanta lo dar siendo dónde mismo ocho seis sera ésos conseguir eran excepto modo voy nunca tendrá suya qeu tres están quiere tenido mío segunda hubo nuestra qué hicieron algunas menos propio les sí hace vamos cuantas unos arribaabajo mios bueno arriba haciendo estuvo vais mismas dicen gueno debido pesar las tienen cuando quienes aunque donde claro aquello podrian que vosotros nueva pueden pasado estan usan uno ambos cualquier esa del fuimos poco tenía dijeron solos eso dio verdadero tu fuera grandes usamos enfrente sido hacerlo cuales ningunos existe junto nos consideró pocos sois algún vuestras largo mia empleas tenemos los delante cuántos realizó era respecto estará ante manifestó porque dejó ampleamos éstas verdadera ahi podrá comentó ésas menudo su intentamos repente dan usted buenos primer tiene estaban muchas supuesto durante día aquélla intentais incluso dentro cual en podrían te hasta por mayor despacio tuyos de trabajamos haya últimas intento estamos suyo ser empleais podrán la le través aquél haceis nadie aproximadamente saber alguna cuánto ti soy raras cierto soyos desde un siempre estos aquellos primero sus van eramos hemos debajo luego también último antano trabaja emplear podrias haber final encima solamente alguno vez podria diferentes otra aquéllas embargo yo aquellas ello contra usas aún mía nuevas da bastante ejemplo casi está partir con éste cuáles tus haces esta vosotras fueron adelante tarde adrede algo uso dijo conocer dos antes ella es aquel informó otros detras ex horas tener lugar han quien tuyo lleva verdad además ellos principalmente proximo trabajan estados paìs total sola entre dieron tampoco cierta dice hacen habrá habia llevar ya cuál estais tanto pronto usa mias son trabajais usais me valor eras apenas tal podriais contigo mas todo mías mientras sabe sé todos cómo estar expresó decir medio éstos será asi consigue afirmó quién conmigo cosas vuestra bajo sobre esas bien estaba parece intentar como pais serán dia tuya pueda intenta toda no cuenta quizas aquella somos suyas creo aqui allí alli este aseguró puede sean empleo primeros tuvo tan habían estas sigue sería próximos misma conseguimos hizo sin poder nuestras sabes buenas diferente buen aun tuyas ir solas primera realizar agregó podemos parte cuántas he añadió salvo siete usar cada estoy propios emplean poca saben próximo ciertas vaya quiza ayer posible pocas ningunas mucho podriamos hablan nuevos pues habla ver más podeis propia peor general gran queremos tambien tenga buena se muchos quedó quizás hacemos mi trabajas segundo nada trabajar cuánta cinco detrás algunos había mucha nuestros señaló mismos veces intentan""".split())
stop["tur"] = set("""<*> . , fakat neye öylemesine çabuk mı bazen nerdeyse onculayın yakından onlara tek cümlemizden bizimki yakinen başkasını birileri dahilen şuna sen birilerini bunları edilecek cümlesini etti şuracıkta epeyi lütfen sizin halihazırda itibarıyla sonunda çokça defa nereye olup kime herkesi kısacası aslında illaki burada bizim etmesi birilerine başkasından bitevi hangisi meğerse evvelemirde birçoğu kendisinden benden oracık dolayısıyla nereden acaba evveli çoğun berikinin cümlemize tam ilgili zarfında bazısına evvelden oluyor hulasaten var birisinde ben evvel geçende buracıkta ettiğini bilcümle burasını gerek geçenlerde yaptığı zaten demin halen ila öbürkü olduğunu böyle nitekim öteki ötekisi bende bile ayrıca hoş çoğundan hiçbirinin de dahi bizzat şunun bize kelli üzere onları sonra bunu karşın tamam haliyle olarak birbirine birden doğru esnasında leh nedense beni elbette bizatihi bizce bunda keşke acep veya hepsinden onlardan bazısının neden senden biraz öbür nerede nazaran insermi birkaçına burasında yenilerde hani neredeyse aynen külliyen ilen kimsecikler hepsini birçok kimisinden etraflı önce evvelce adamakıllı nice onların birkez međer yapacak beri cümlemizi yakında kendi pekçe ettiği kendilerini zira hariç ona oradan sonradan burasının birice oranla olursa yaptı emme ise şunu filancanın çoklarınca ora rağmen şura tamamıyla dolayı gelgelelim birinin birşeyi kala iyice birilerinin yalnızca ne itibaren hiç bazı hasebiyle dek değil diğeri öz kiminin çoğu nere az önceden bunun gayrı gibilerden olan birkaçında sadece beriki birlikte kimisine hiçbirinden burasından yani birçoğunu hakeza inen sanki öbüründe itibariyle orada sahiden iş birkaç öylelikle elbet gah velhasıl berisi sahi ki tabii artık kendisinin bizden diye vasıtasıyla bunların kimden birdenbire birkaçını birçoğunun denli bari yalnız işte çoğunun ister yapılan birkaçı nedenler birisinden böylelikle olduklarını esasen şundan şurası ile vardı iken naşi birkaçından diğerine çabukça daima diğerinden epey birbirinde pek bazısında önceleri yapılması birine birçoğuna daha şunlar sana açıkça maada enikonu tarafından hep senin biteviye hiçbir veyahut çeşitli illa evvela halbuki yüzünden niye olsa ya birisi hepsinde derakap yahut hatta çoğunu gayri birbiri sizi çoğunlukla meğerki mü öyle kendisi nedeniyle oldukça böylecene kimisinde derhal yine sonraları bunlar neyi ancak bu birinde kendisine kez ama demincek şeyler hele niçin oraya açıkçası yapmak dair ederek şunda ait hiçbirinde pekala onlar hangisinden başkası derken herkesin dahası hangi şimdi cümlesi olması nedenle kimse çok herkesten birbirinin hiçbirini oysaki edilmesi birçoğundan hepsinin iyicene başkasında başkasına herhangi kaçına gine herkes nerde birşey benim şeyden kimisi evleviyetle en oranca olsun mademki kaçından onu bundan birkaçının kaçında seni gene tüm onca boşuna kendisinde yapıyor filanca siz lakin gerçi mamafih handiyse birbirinden nerden olduğu bazısından çokluk çoğunca madem hasılı hem buradan berikiyi kim için şu hangisine çoğunda öbüründen deminden kimisinin kaçını böylesine kendilerinde kaçı olmak neresi yerine eğer birilerinde nasıl kendilerine cümlesinden eden birçoğunda birisini bizcileyin mebni cuk ediyor nedenlerden kaffesi birini yaptığını ondan bizi biri meğer edecek epeyce imdi öbürünü neyse birisinin velhasılıkelam başka onun şayet dayanarak ilk nihayetinde gırla anca çoklukla etraflıca şunların onda büsbütün şöyle gayet oysa gibisinden bazısı gayetle kaçının peki gibi arada kere indinde böylemesine başkasının ve burasına kendilerinden belki velev göre birbirini dahil mi hiçbiri yoksa şuracık binaenaleyh kimsecik burası kısaca kezalik birinden kendini öylece buna kaynak çünkü o iyi yoluyla diğer bütün zati yeniden hepsi hangisinde böylece kendisini nasılsa adeta keza kimi yaptıkları mu öncelikle birisine öbürüne peyderpey şey olur şuncacık nihayet netekim bana bittabi yakınlarda biz cümlesine binaen bazısını oldu sizden ediliyor diğerini birilerinden birazdan değin tamamen henüz şeyi kendilerinin kadar hiçbirine kah cümlesinin şunları kanımca oracıkta yok da her amma öbürü çokları""".split())
stop["zho"] = set("""<*> 二话不说 今天 再其次 同时 几时 本人 此地 蛮 恰恰相反 ［②①］ 屡次 怕 挨次 // 自个儿 其他 从速 一样 最高 那般 近来 从宽 由是 ［②ａ］ 全身心 替 首先 普通 = 不能 反之 反倒是 ＋＋ 自身 坚持 本 顷刻 它 可以 为什么 不惟 总而言之 藉以 每时每刻 将 # ' 这儿 怎 相对 ℃ 谨 小 ［①⑨］ 这样 某某 得出 ‘ 各 :: 遭到 当着 ＿ 趁势 双方 如何 各级 前者 当儿 从早到晚 与 〔 倒不如说 何时 刚巧 宁肯 年复一年 豁然 真是 前后 一边 构成 所在 必 复杂 彼时 ［⑤ａ］ 你的 必然 显然 几 如上所述 目前 应该 没有 即若 大面儿上 Ψ 反之亦然 综上所述 倒不如 乘机 间或 呆呆地 则甚 以来 例如 基本上 除却 Ａ 抑或 代替 一方面 差一点 该 岂止 什么 无论 已 挨门逐户 也 加以 需要 ｝ 居然 屡次三番 的 仍 他 遵照 照着 ［②Ｂ］ 不怎么 6 ］［ ］ 加入 既往 每天 大事 而况 那样 长期以来 ’‘ 被 打开天窗说亮话 个人 有 还要 ￥ 凡 二 无宁 同 ［⑥］ ——— 进步 上面 大举 （ 乃至于 任凭 老老实实 ［①Ｅ］ 8 连日来 过于 安全 使得 ８ 哗 或是 内 论 成年累月 其二 2 是不是 这麽 ＜ 一定 ［①①］ 接连不断 人人 然後 比照 这个 非但 漫说 γ 嘻 分期 基本 . 怪不得 绝对 ５：０ 不拘 毫无保留地 的话 设使 _ 半 该当 > 上 喀 社会主义 到了儿 大多 这些 要是 如上 抽冷子 反之则 还是 您是 且说 了 不免 地 若非 要不 历 ［③ｂ］ 欢迎 这就是说 挨着 有的是 二话没说 向使 以致 您们 简而言之 ［②③］ 扑通 两者 甚么 :// 不仅...而且 竟 加上 常 5 出来 如是 反映 来自 决非 那么样 据实 看样子 由于 密切 不对 怎奈 近 大略 7 不得了 嘿 再则 以至于 哪天 别人 纵令 设或 据称 庶乎 并不是 敢于 方面 相同 得了 按 不可 丰富 把 可能 忽地 望 而又 清楚 ［④］ 多年前 对比 不至于 应用 简直 一些 产生 认识 由此 归 有所 〉 按期 与否 高低 先不先 不仅 如期 末##末 尽早 趁热 往往 相似 何 ................... 普遍 ▲ 之所以 像 存心 ＊ 零 大抵 达到 保险 赶 特殊 惟其 上下 啊呀 实现 本地 很 过来 不日 … 逐步 累年 巴巴 长此下去 跟 本身 尔 一 您 连 ［⑦］ 从今以后 己 毫无 自打 怎么 毫无例外 当场 各地 合理 并 并不 不已 故 恰似 不止 立刻 所 继而 不单 切切 ［②ｄ］ 冒 再说 陈年 突出 因着 左右 * 任务 联袂 常言道 能否 ［③ｄ］ 即将 叮咚 怎么办 如此等等 ［①ｉ］ 屡 结合 到底 １２％ 最大 那个 若夫 呐 越是 不得不 ...... －β 昂然 ⑦ 以及 嗯 坚决 ~~~~ 嘘 三番两次 以为 分别 进行 [ 不但...而且 因 不胜 如 自 第二 固然 缕缕 以期 开展 不起 ［②ｊ］ 嗡 元／吨 傥然 到处 某 犹自 组成 怎麽 ５ 十分 赶快 一番 ? 吧哒 吗 尽心尽力 何乐而不为 方 果然 这点 又及 此间 别的 挨个 一面 → 隔夜 隔日 这一来 适当 起首 方能 默默地 若果 ⑤ 那儿 ７ 高兴 多多 当地 ＋ξ Δ 依 又 ［②ｈ］ 反过来 古来 ［②②］ 老 ＆ 过去 默然 精光 ［①Ａ］ ③ 根本 不常 举行 略 何止 毋宁 一时 认为 偶而 叮当 从未 孰知 暗自 较 难道说 逢 归齐 之一 有着 不一 庶几 ` 别说 或曰 难得 进而 遵循 只有 开始 一个 照 既然 显著 不变 倘若 使用 略加 莫若 放量 乘虚 难怪 另一方面 ［⑤］］ 近年来 / 所幸 不问 ； sub ＞ 局外 绝非 及其 打 将才 说明 & 我的 千 均 这会儿 】 即令 是以 打从 除此以外 ④ 基于 看 ＺＸＦＩＴＬ 这边 部分 宁愿 颇 ＜＜ 呵呵 就是 怎样 不择手段 砰 起来 类如 立马 嘿嘿 皆可 从严 岂非 连日 来不及 连同 当前 ： ［①⑤］ 因而 连连 ［④ａ］ 后 不少 通常 少数 乃至 成心 相反 从事 几乎 多多少少 φ． ［⑨］ 极其 者 前进 之前 周围 假若 甚至 串行 那么些 练习 ［①］ 单纯 喂 常言说得好 嘎登 较之 》 不比 比起 主张 甭 对于 勃然 从小 在下 μ 受到 尚且 容易 啐 顿时 尽心竭力 汝 不力 其次 咚 大约 随著 依靠 哈哈 得起 看出 ａ］ 老是 初 吓 何况 一下 正值 В 继后 某些 任 呜 使 假使 啪达 任何 保管 猛然间 ［①⑦］ 不然的话 倘使 哪个 然后 之 另一个 倒是 别是 赶早不赶晚 当中 来着 向着 挨门挨户 要求 换句话说 比及 并且 通过 全体 平素 不光 ［②ｂ］ 范围 『 非得 嗳 ～＋ 一致 随 取道 >> 偶尔 得 上升 本着 六 哇 成年 率尔 匆匆 不经意 哪 趁早 等等 那么 比如说 会 看起来 共总 针对 ［①ｃ］ 进去 现在 ∪φ∈ 前此 嘎嘎 正如 大批 正在 纵然 ［①③］ 比如 没 或 次第 莫非 维持 ＇ 这里 ′∈ 何妨 只要 不可抗拒 呵 趁 经过 长线 鄙人 别处 据此 自己 就要 她 呼啦 具体来说 不 及 哪儿 除了 好 够瞧的 依据 ⑧ 哪年 嘛 弹指之间 光 如常 注意 乘胜 除 今后 在于 整个 嗬 不外 ｂ］ 顺 不独 何苦 为着 极端 当时 方才 穷年累月 觉得 继续 况且 接下来 : 全部 所有 反而 亲口 连声 沿 略微 我 ＜λ 转动 .. ＜± 来看 何以 与此同时 出于 ［②⑤］ 余外 不同 焉 哎呀 继之 表示 ＝［ 极了 自家 偏偏 下 极大 仍旧 就是了 ［①④］ 竟而 边 已矣 从不 来 吱 ［－ 宣布 矣乎 诸 莫不 互相 断然 孰料 有力 就地 它是 看上去 反倒 那时 要 不管 既 顷 立 造成 ～ - 不尽然 今 亦 接着 不特 ① 极力 哼 归根到底 种 致 敞开儿 独 当即 迟早 而是 毕竟 更加 以前 绝顶 离 方便 此外 着 得天独厚 ｆ］ 重新 好的 而言 按时 以至 转贴 便于 ［①Ｂ］ ０：２ 不止一次 并排 纵 曾 全力 满 腾 只 另方面 不敢 一旦 ［①Ｃ］ 所谓 顺着 从重 再者 我们 如今 经常 掌握 管 不定 保持 立即 尽然 喏 企图 总的来说 ［④ｃ］ 单 多亏 顷刻之间 .数 难道 逐渐 限制 尽管如此 诸如 却不 率然 ［①②］ 省得 ［①ｏ］ 传说 故而 非特 之後 ′｜ 绝不 从新 不下 非常 正常 这 巴 巩固 何必 失去 很多 嘎 不然 恐怕 设若 』 谁 不了 些 具有 良好 他人 看看 对应 似乎 纯 无 下列 不满 乘势 说说 所以 上来 。 另外 唯有 不要 哗啦 欤 并肩 到 迅速 还 必须 先生 伙同 动不动 才能 以便 已经 一起 宁可 ＝″ — 当庭 将近 呕 ［③Ｆ］ 不足 成为 莫 由 防止 按理 个 运用 ［⑤］ 每逢 定 ［③］ 但 理该 常常 截至 她的 愿意 数/ 似的 一则 争取 一则通过 ［③⑩］ 彻底 召开 ＝☆ 好象 当真 三天两头 八成 话说 % 避免 战斗 特别是 用 你们 惯常 ── 一天 具体 并非 仅仅 一何 传闻 极度 朝 我是 给 儿 具体地说 满足 能够 自从 呼哧 ［②⑩］ 还有 从而 论说 哪些 当头 着呢 据 之类 于 至 更 不大 与其说 必定 赖以 ［④ｅ］ 今年 大家 它们的 借此 ^ 不尽 差不多 若 往 咳 什么样 .一 却 开外 待 ［①ｇ］ 见 深入 屡屡 暗地里 不仅仅是 ( 岂但 ３ 恰恰 固 然而 恰逢 完全 为了 亲身 尽量 相等 矣哉 举凡 总的来看 诸位 都 以 较为 故此 ４ 倍加 云云 这次 其 现代 …… 可 确定 带 强烈 那麽 有效 起初 日臻 大体上 不但 殆 不若 一. 人 反手 除此之外 更进一步 能 Ⅲ 反过来说 ［② 表明 譬如 仍然 ! 即便 广泛 根据 ％ 切莫 不仅仅 只是 看到 规定 五 全年 ｎｇ昉 ２ 敢 除非 这么点儿 先後 ××× 倘或 或者 可是 乘 因此 是的 几度 好在 借以 最後 哩 陡然 用来 ｝＞ ＝（ 采取 来得及 ． 共 Ｒ．Ｌ． ［②④ 于是乎 公然 对方 臭 而外 大大 咧 光是 明显 理当 甚或 虽则 ［⑧］ ［④ｄ］ 以外 ” 靠 再者说 莫不然 哪怕 ＠ 大致 附近 总是 而后 啊哟 ＞λ 不会 尔等 如若 多多益善 什麽 交口 Lex ！ 后面 而已 决定 过 某个 冲 相当 从古到今 甚且 不妨 诚然 啥 彼此 ［③ｇ］ ... 即或 遇到 对 理应 各个 上去 不必 旁人 适应 ［①⑧］ 碰巧 总结 阿 老大 重大 一次 不亦乐乎 愤然 ｛－ 每 不怕 七 格外 自各儿 极为 及时 出现 分期分批 不过 暗中 A 〈 哈 以後 没奈何 相信 具体说来 除去 exp 纵使 她们 总的说来 有点 ’ 为何 —— 再 或多或少 略为 如果 咋 哪边 了解 呗 多次 , 临 尽可能 转变 三 贼死 刚 沙沙 出 竟然 ）、 ｜ 几番 大量 上述 ［ 原来 也就是说 此时 且 后来 既...又 姑且 即刻 紧接着 ［①Ｄ］ 3 权时 另 同一 亲手 从古至今 意思 为此 相应 恰好 有的 ③］ 然则 ［②ｉ］ 如此 不限 ［②⑥］ 这么样 或许 ［②］ + 俺 万一 不够 以上 当下 而 .日 谁料 ./ ［③ａ］ 这种 齐 下去 达旦 叫做 哼唷 恍然 1 趁机 大多数 则 是 奋勇 ／ 起先 不料 } 挨家挨户 据悉 相对而言 严格 为止 准备 据说 凭借 比 」 别管 後面 ｛ -- 到目前为止 另悉 立时 日复一日 联系 别 全然 啷当 最近 自后 依照 喽 心里 迄 呸 ［⑤ｄ］ 请勿 不巧 不可开交 ２．３％ 扩大 各种 形成 随后 知道 ］∧′＝［ 和 ［②ｇ］ 作为 尔后 ［①ａ］ 来说 策略地 就算 届时 加之 专门 替代 ＃ 或则 ）÷（１－ 而且 按照 切 考虑 每每 即如 各式 ■ 既是 ”， ＋ 云尔 那里 啊 》）， 甚至于 【 ［①ｅ］ 假如 真正 非独 此次 集中 互 不久 倘 从此以后 更为 ［⑤ｆ］ 以免 就是说 距 处处 急匆匆 ［⑩］ 随时 时候 呢 三番五次 ︿ 接著 全面 除此而外 ，也 有关 连袂 嗡嗡 甚而 巨大 切不可 此处 做到 直接 @ ② 一片 咦 他是 ∈［ 简言之 ［②⑧］ 不外乎 不知不觉 共同 单单 毫不 0 换言之 结果 起 不只 经 " 独自 每个 顷刻间 强调 一一 待到 刚才 即使 川流不息 － 忽然 人民 反应 兮 先后 如下 恰如 并没有 咱 －－ 哪样 到头 起头 那边 归根结底 乎 | 乘隙 出去 那会儿 他的 看来 粗 对待 此中 让 ［②⑦］ 累次 其它 ［③ｈ］ 倘然 彻夜 各人 个别 牢牢 饱 起见 你 谁人 ＬＩ 不迭 要么 ＜φ 其余 迫于 取得 除外 实际 大凡 各位 认真 许多 白 …………………………………………………③ 长话短说 朝着 移动 近几年来 但是 但愿 允许 问题 必要 然 ; 纯粹 ＝－ 么 $ 决不 比方 最 後来 有利 此 譬喻 ［④ｂ］ 恰巧 顶多 中小 sup 其后 这时 到头来 ， 去 日渐 除此 下来 中间 切勿 般的 始而 至于 二来 何尝 风雨无阻 ［②ｆ］ 多少 梆 ［①ｈ］ 其实 ［③ｃ］ 充分 里面 ＝｛ ［①ｄ］ 大 大力 明确 伟大 当 的确 诚如 从头 ｃ］ 截然 概 当然 以故 怎么样 倍感 也好 × 适用 大不了 一转眼 ［①ｆ］ 且不说 九 无法 宁 是否 尽 以后 只当 凝神 与其 哦 窃 不得已 等到 从优 为什麽 特点 ？ 行动 背靠背 快 从轻 何处 敢情 有及 呀 按说 虽然 即 等 积极 因为 凑巧 便 借 可好 至今 尔尔 不由得 多么 后者 如次 从中 拿 甫 快要 亲眼 咱们 变成 人家 凡是 ＝ 若是 乒 尽快 不论 ⑨ 获得 下面 促进 ［②ｃ］ 何须 拦腰 φ 背地里 ９ 在 ⑥ 多年来 趁便 较比 这么 鉴于 呜呼 也是 前面 说来 ) 至若 啦 据我所知 ㈧ 这么些 不时 它的 不曾 否则 凭 罢了 进入 从此 ＜Δ 尤其 非徒 日益 但凡 如其 ［②Ｇ］ 如前所述 仅 有时 绝 乃 她是 4 ［②ｅ］ ～± 这般 直到 叫 为主 ≈ 那些 今後 关于 正巧 ６ 吧 兼之 亲自 从 啊哈 它们 怪 人们 彼 尽如人意 那 来讲 为 ［＊］ １ 〕 看见 活 岂 不是 千万 八 不消 以下 即是说 ［⑤ｂ］ 临到 不断 及至 之后 向 呃 可见 才 突然 、 ｅ］ 分头 猛然 ⑩ 比较 不得 多 《 ［］ 充其极 不管怎样 全都 从无到有 只怕 大体 于是 “ 果真 立地 大概 瑟瑟 加强 大都 一来 ] 再有 ＄ < ０ 喔唷 ） ［⑤ｅ］ 得到 极 趁着 不如 而论 犹且 ↑ 你是 每年 ［③ｅ］ 千万千万 也罢 矣 属于 哪里 动辄 四 故意 慢说 并无 严重 正是 就此 随着 各自 如同 ②ｃ 重要 总之 虽说 只限 应当 主要 刚好 一般 不成 处理 同样 帮助 弗 他们 一切 其一 广大 最好 介于 就 当口儿 哟 哎哟 进来 9 · 马上 存在 哎 －［＊］－ 乌乎 完成 由此可见 日见 多数 除开 常言说 俺们 唉 很少 第 曾经 哉 另行 从来 有著 〕〔 传 充其量 几经 再次 行为 要不是 沿着 引起 白白 并没 ［③①］ 轰然 因了 眨眼 究竟 不再 们 每当 不能不 要不然 必将 那末 谁知 １． 此后 ［①⑥］ 处在 大张旗鼓 虽 路经 将要 尽管 ~ 最后 只消 奇 一直 莫如 其中 有些 奈 难说""".split())
stop['fas'] = set("""خاص آوری اینجا آورد لذا کلی سوم دانست داشتند فقط کامل نه جمع پنج بودن بعد می‌شوند نیاز نشست همین شوند خوبی ایشان بلکه یابد علاوه پر کرده جدی کافی باره نیستند شاید یا پخش هایی کدام طبق حال منظور همان شش ع یکدیگر آمده می‌توان مناسب اجرا همچون اینکه باید صرف گونه اگر می‌کند به تاکنون خویش پیش ما اول داشته‌باشد کنید دیگری آنچه زیرا آقای رشد که دو وارد غیر خودش نیز سپس حتی میان دچار باز گرفته‌است ندارند هنوز اولین آمد ساله ناشی رفت بیشتر بهترین جاری برخی چیزی زیر گردد می اخیر یافته‌است خاطرنشان دور نسبت شروع بخش می‌یابد روبه آخرین گاه سهم دوم هم تبدیل تو بوده بدون درباره فرد کنندگان بالا ساز شده قابل داریم حالا کردم بر هر نوعی همیشه و نیست بوده‌است تمام کوچک برابر روند خواهد‌کرد روش می‌کردند بیش در بین جز من تعداد بیان متفاوت خیلی زیادی طول خود ضمن دیگر گرفته کسی آنجا دارای همه بعضی از بندی می‌رود عالی جمعی حد قبل خوب شده‌اند همواره مهم می‌توانند می‌دهند داشته بی سالهای متاسفانه وی ولی بسیار آنان زاده نظر دهه سه کمی لحاظ شود بیشتری می‌شود راه داده همچنین دارد جدید تا محسوب کنم گذاری سبب باشد حل نمی‌شود سوی شمار داده‌است علت او بروز موجب دیگران شان مربوط بود کنار سعی باعث می‌رسد نیمه سراسر می‌آید افرادی افزود رو کرده‌بود چند کننده روی تحت بیرون چون شده‌است نباید اثر براساس بخشی داشتن ابتدا می‌تواند کردند پیدا یک کاملا حدود اغلب گفته چهار تعیین نوع اش امر پی را طور بهتر دهد جایی یعنی شدند دادن آمده‌است اکنون تنها عدم گرفت داشته‌است مورد می‌کنند جای یافته همچنان شامل می‌کرد آن می‌گویند ندارد بسیاری گفت بودند کرده‌است هستند چرا نحوه ریزی گروهی پشت مواجه گیری سمت کند کنند سال‌های هنگام لازم دارند چگونه چه پس دسته حداقل سی کرده‌اند فوق متر مانند نبود برای می‌شد فردی ترتیب علیه شما می‌باشد مشخص داد افراد وجود شده‌بود سازی طی البته است کم کنیم گیرد می‌کنیم با رسید جا کردن شدن می‌دهد کل خصوص دهند هستیم نزدیک درون آیا طرف بزرگ این دار دوباره تمامی بار یافت ممکن خواهد‌شد چهارم کرد اند داشته‌باشند کسانی اما می‌گیرد مثل زیاد حالی ویژه شد شخصی باشند عهده خطر یکی سایر نخستین می‌گوید جریان مدت تهیه نظیر بنابراین چیز وگو فکر داشت عین دادند چنین برداری تغییر می‌کنم وقتی امکان نخست هیچ رسیدن خواهد‌بود آنها آنکه """.split())


def get_genre(corpus, doc):
    if corpus in ["eng.rst.rstdt","eng.pdtb.pdtb","deu.rst.pcc","por.rst.cstn","zho.pdtb.cdtb"]:
        return "news"
    if corpus == "rus.rst.rrt":
        if "news" in doc:
            return "news"
        else:
            return doc.split("_")[0]
    if corpus == "eng.rst.gum":
        return doc.split("_")[1]
    if corpus == "eng.rst.stac":
        return "chat"
    if corpus == "eus.rst.ert":
        if doc.startswith("SENT"):
            return doc[4:7]
        else:
            return doc[:3]
    if corpus == "fra.sdrt.annodis":
        if doc.startswith("wik"):
            return "wiki"
        else:
            return doc.split("_")[0]
    if corpus in ["nld.rst.nldt","spa.rst.rststb"]:
        return doc[:2]
    if corpus in ["spa.rst.sctb","spa.rst.sctb"]:
        if doc.startswith("TERM"):
            return "TERM"
        else:
            return doc.split("_")[0]
    if corpus == "fas.rst.prstc":
        return doc[0:6]
    print("Unknown corpus: " + corpus)
    return "_"


def get_head_info(unit_span, toks):
    parts = unit_span.split(",")
    covered = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            start = int(start)
            end = int(end)
            covered += list(range(start,end+1))
        else:
            covered.append(int(part))

    head = toks[covered[0]]
    head_dir = "ROOT"
    for i in covered:
        tok = toks[i]
        if tok.head == 0:
            head = tok
            break
        if tok.head < covered[0]:
            head = tok
            head_dir = "LEFT"
            break
        if tok.head > covered[-1]:
            head = tok
            head_dir = "RIGHT"
            break
    return head.deprel, head.xpos, head_dir


def get_case(text,lang):
    stopwords = stop[lang]
    words = text.split()
    non_stop = [w for w in words if w not in stopwords]
    if all([w[0].isupper() for w in non_stop]):
        return "title"
    elif words[0][0].isupper():
        return "cap_initial"
    else:
        return "other"


def process_relfile(infile, conllu, corpus, as_string=False, keep_all_columns=False, do_zscore=False):
    lang = corpus.split(".")[0]
    try:
        stop_list = stop[lang]
    except:
        stop_list = []

    if not as_string:
        infile = io.open(infile,encoding="utf8").read().strip()
        conllu = io.open(conllu,encoding="utf8").read().strip()

    unit_freqs = defaultdict(int)
    unit_spans = defaultdict(set)
    lines = infile.split("\n")
    # Pass 1: Collect number of instances for each head EDU and sequential ID
    for i, line in enumerate(lines):
        if "\t" in line and i > 0:  # Skip header
            cols = {}
            for j, col in enumerate(line.split("\t")):
                cols[headers[j]] = col
            if cols["dir"] == "1>2":
                head_unit = cols["unit2_toks"]
            else:
                head_unit = cols["unit1_toks"]
            unit_freqs[(cols["doc"], head_unit)] += 1
            unit_spans[cols["doc"]].add(cols["unit1_toks"])
            unit_spans[cols["doc"]].add(cols["unit2_toks"])

    units2start = defaultdict(dict)
    for doc in unit_spans:
        for span in unit_spans[doc]:
            start = re.split(r'[-,]',span)[0]
            units2start[doc][span] = int(start)
    unit_order = defaultdict(dict)
    for doc in units2start:
        counter = 0
        for u in sorted(units2start[doc],key=lambda x: units2start[doc][x]):
            unit_order[doc][u] = counter
            counter += 1

    # Pass 2: Get conllu data
    tokmap = defaultdict(dict)
    toknum = 1
    offset = 0
    speaker = "none"
    docname = None
    fields = [0]
    for line in conllu.split("\n"):
        if "# newdoc" in line:
            docname = line.split("=")[1].strip()
            toknum = 1
            offset = 0
        if "# speaker" in line:
            speaker = line.split("=")[1].strip()
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0] or "." in fields[0]:
                continue
#             head = 0 if fields[6] == "0" else int(fields[6]) + offset
            head = 0 if fields[6] in ["0", "", "_"] else int(fields[6]) + offset
            tok = Token(toknum, fields[0], fields[4], head, fields[7], speaker)
            tokmap[docname][toknum] = tok
            toknum += 1
        if len(line) == 0:
            offset += int(fields[0])

    output = []
    # Pass 3: Build features
    for i, line in enumerate(lines):
        if "\t" in line and i > 0:  # Skip header
            feats = {}
            for j, col in enumerate(line.split("\t")):
                feats[headers[j]] = col
            if feats["dir"] == "1>2":
                head_unit = feats["unit2_toks"]
                child_unit = feats["unit1_toks"]
            else:
                head_unit = feats["unit1_toks"]
                child_unit = feats["unit2_toks"]
            feats["nuc_children"] = unit_freqs[(feats["doc"],head_unit)]
            feats["sat_children"] = unit_freqs[(feats["doc"],child_unit)]
            feats["genre"] = get_genre(corpus,feats["doc"])
            feats["u1_discontinuous"] = "<*>" in feats["unit1_txt"]
            feats["u2_discontinuous"] = "<*>" in feats["unit2_txt"]
            feats["u1_issent"] = feats["unit1_txt"] == feats["unit1_sent"]
            feats["u2_issent"] = feats["unit2_txt"] == feats["unit2_sent"]
            feats["u1_length"] = feats["unit1_txt"].replace("<*> ","").count(" ") + 1
            feats["u2_length"] = feats["unit2_txt"].replace("<*> ","").count(" ") + 1
            feats["length_ratio"] = feats["u1_length"]/feats["u2_length"]
            u1_start = re.split(r'[,-]',feats["unit1_toks"])[0]
            u2_start = re.split(r'[,-]',feats["unit2_toks"])[0]
            feats["u1_speaker"] = tokmap[feats["doc"]][int(u1_start)].speaker
            feats["u2_speaker"] = tokmap[feats["doc"]][int(u2_start)].speaker
            feats["same_speaker"] = feats["u1_speaker"] == feats["u2_speaker"]
            u1_func, u1_pos, u1_depdir = get_head_info(feats["unit1_toks"], tokmap[feats["doc"]])
            u2_func, u2_pos, u2_depdir = get_head_info(feats["unit2_toks"], tokmap[feats["doc"]])
            feats["u1_func"] = u1_func
            feats["u1_pos"] = u1_pos
            feats["u1_depdir"] = u1_depdir
            feats["u2_func"] = u2_func
            feats["u2_pos"] = u2_pos
            feats["u2_depdir"] = u2_depdir
            feats["doclen"] = max(tokmap[feats["doc"]])
            feats["u1_position"] = 0.0 if u1_start == "1" else int(u1_start) / feats["doclen"]  # Position as fraction of doc length
            feats["u2_position"] = 0.0 if u2_start == "1" else int(u2_start) / feats["doclen"]  # Position as fraction of doc length
            feats["percent_distance"] = feats["u2_position"] - feats["u1_position"]  # Distance in tokens as fraction of doc length
            # Distance in ordered spans (EDUs, or for PDTB how many other units are attested in between):
            feats["distance"] = unit_order[feats["doc"]][feats["unit2_toks"]] - unit_order[feats["doc"]][feats["unit1_toks"]]
            unit1_words = feats["unit1_txt"].split(" ")
            unit2_words = feats["unit2_txt"].split(" ")
            overlap_words = [w for w in unit1_words if w in unit2_words and w not in stop_list]
            feats["lex_overlap_words"] = " ".join(sorted(overlap_words)) if len(overlap_words) > 0 else "_"
            feats["lex_overlap_length"] = feats["lex_overlap_words"].count(" ") + 1 if len(overlap_words) > 0 else 0
            feats["unit1_case"] = get_case(feats["unit1_txt"],lang)
            feats["unit2_case"] = get_case(feats["unit2_txt"],lang)
            if not keep_all_columns:
                del feats["unit1_sent"]
                del feats["unit1_toks"]
                del feats["unit1_txt"]
                del feats["s1_toks"]
                del feats["unit2_sent"]
                del feats["unit2_toks"]
                del feats["unit2_txt"]
                del feats["s2_toks"]
                del feats["label"]
                del feats["orig_label"]
                del feats["doc"]

            output.append(feats)

    if do_zscore:
        for featname in [
            "doclen",
            "u1_position",
            "u2_position",
            "distance",
            "nuc_children",
            "sat_children",
            "lex_overlap_length"
        ]:
            xs = [x[featname] for x in output]
            new_xs = zscore(xs)
            for o, x in zip(output, new_xs):
                o[featname] = x.item()

    return output


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--disrpt_data", action="store", default=".."+os.sep+"repo"+os.sep+"data"+os.sep,help="directory with DISRPT repo data folder")
    p.add_argument("-c","--corpus", default="eng.rst.gum", help="corpus name")

    opts = p.parse_args()

    if not opts.disrpt_data.endswith(os.sep):
        opts.disrpt_data += os.sep

    corpus = opts.corpus
    corpus_root = opts.disrpt_data + corpus + os.sep
    files = glob(corpus_root + "*.rels")

    for file_ in files:
        rows = process_relfile(file_, file_.replace(".rels",".conllu"), corpus, keep_all_columns=True)

        output = []
        ordered = []
        for row in rows:
            all_keys = row.keys()
            header_keys = [k for k in headers if "label" not in k]
            other_keys = [k for k in row if "label" not in k and k not in headers]
            ordered = header_keys + other_keys + ["label"]
            out_row = []
            for k in ordered:
                if isinstance(row[k],float) and len(str(row[k]))>3:
                    out_row.append("{:.3f}".format(row[k]).rstrip('0'))
                else:
                    out_row.append(str(row[k]))
            output.append("\t".join(out_row))

        output = ["\t".join(ordered)] + output

        #print ("\n".join(output[:100]))
        #quit()
        with io.open(os.path.basename(file_).replace(".rels","_enriched.rels"),'w',encoding="utf8",newline="\n") as f:
            f.write("\n".join(output)+"\n")
