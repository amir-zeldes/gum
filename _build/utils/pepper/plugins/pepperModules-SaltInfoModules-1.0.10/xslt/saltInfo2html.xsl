<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="2.0">
    
    <!-- first output in html format, use xhtml for wellformedness of unary tags -->
    <xsl:output encoding="UTF-8" indent="yes" method="xhtml" doctype-system="about:legacy-compat"/>
    
    <!-- second output file for information saved in json format -->
    <xsl:output method="text" indent="no" name="json" encoding="UTF-8"/>
    
    <!-- third output file for main page -->
    <xsl:output method="html" indent="yes" name="main" encoding="UTF-8"/>
    
    <!-- fourth output file for customization infos -->
    <xsl:output method="text" indent="no" name="customization" encoding="UTF-8"/>
    
    <!-- set the minimum of annotations shown at the tables if uncollapsed -->
    <xsl:variable name="minNumOfAnnos">5</xsl:variable>
    
    <!-- set the number of tooltips for meta data and annotations -->
    <xsl:variable name="NumOfTooltips">3</xsl:variable>
    
<!-- set createJsonForAllAnnos to "true", if all annotations shall be loaded into json, even those with less than 5 values -->
    <xsl:variable name="createJsonForAllAnnos" select="false()" />
    
    <!-- check if file is main corpus -->
    <xsl:variable name="isMainCorpus" select="sCorpusInfo/@sName=$corpusname"/>
    
    <!-- get corpus or document name from id and save it as a variable for later use -->
    <xsl:variable name="corpusname">
        <xsl:choose>
            <!-- if current file is a subcorpus or document extract name from id by selecting the string between first and second slash -->
            <xsl:when test="string-length(root()/node()/@id) - string-length(replace(root()/node()/@id, '/', '')) > 1">
            <xsl:value-of select="substring-before(substring-after(root()/node()/@id, 'salt:/'),'/')"/>
        </xsl:when>
            <xsl:otherwise>
                <!-- if current file is main corpus extract name from id -->
                <xsl:value-of select="substring-after(root()/node()/@id, 'salt:/')"/>
            </xsl:otherwise>
    </xsl:choose>
    </xsl:variable>
    
    <!-- extract name of current file from id by selecting string after last slash -->
    <xsl:variable name="currentFile">
        <xsl:value-of select="replace(root()/node()/@id,'.*/','')"></xsl:value-of>
    </xsl:variable>
   
    <!-- define output name for json file -->
    <xsl:param name="jsonOutputName">./anno_<xsl:value-of select="$currentFile"/>.json</xsl:param>
    <xsl:variable name="jsonOutputPath">./<xsl:value-of select="substring-after(replace(root()/node()/@id, $currentFile, concat('anno_', $currentFile, '.json')), 'salt:/')"/></xsl:variable>
    
    <!-- tooltip descriptions for structural elements -->
    <xsl:variable name="SNode">Total number of nodes in the current document or corpus. An SNode is an abstract node which could be instantiated as e.g. SToken, SSpan, SStructure, STextualDS and so on.</xsl:variable>
    <xsl:variable name="SRelation">Total number of  relations in the current document or corpus. An SRelation is an abstract relation which could be instantiated as e.g. STextualRelation, SSpanningRelation, SDominanceRelation and so on.</xsl:variable>
    <xsl:variable name="SSpan">Number of spans in the current document or corpus. A span bundles between 1 and n tokens to a set.</xsl:variable>
    <xsl:variable name="SSpanningRelation">Number of relations in the current document or corpus to connect spans (SSpan) with tokens (SToken).</xsl:variable>
    <xsl:variable name="STextualDS">Number of relations in the current document or corpus to connect a token (SToken) with a textual data source (STextualDS).</xsl:variable>
    <xsl:variable name="STimeline">In Salt, a node common timeline is used, to bring tokens into a chronological order. For instance to identify if one token corresponding to one text occurs before or after another token corresponding to another text. This is important to model dialogue corpora.</xsl:variable>
    <xsl:variable name="SToken">Number of tokens in the current document or corpus. A token in Salt is the smallest annotatable unit of text and has no linguistic meaning. A token could be a character, a syllable, a word, a sentence etc.</xsl:variable>
    <xsl:variable name="SPointingRelation">Number of relations in the current document or corpus with an underspecified linguistic meaning. A SPointing  relation can connect nodes like SToken, SSpan and SStructure with each other to model for instance anaphoric relations.</xsl:variable>
    <xsl:variable name="STextualRelation">Number of relations in the current document or corpus to connect a token (SToken) with a textual data source (STextualDS).</xsl:variable>
    <xsl:variable name="SStructure">Number of hierarchical structures in the current document or corpus. SStructure objects in Salt are used to represent hierarchies e.g. for constituents.</xsl:variable>
    <xsl:variable name="SDominanceRelation">Number of relations in the current document or corpus to connect hierarchical nodes (SStructure) with other SStructure, SSpan or SToken objects.</xsl:variable>
    <xsl:variable name="SOrderRelation">Number of relations in the current document or corpus to order SNode objects. This class of relations is used to manage conflicting token levels as they can occur for instance in dialogues or historic texts (with several primary texts like transcription, diplomatic transcription, normalization etc.).</xsl:variable>
    <xsl:variable name="STimelineRelation">Number of relations in the current document or corpus to connect a token (SToken) with the common timeline (STimeline).</xsl:variable>

	<!-- tooltips for functional buttons -->    
    <xsl:variable name="TOOLTIP_DOWNLOAD">Downloads annotation values and corresponding frequency as CSV file (you need to expand the view to download all values)</xsl:variable>
    <xsl:variable name="TOOLTIP_BOXES">Draws boxes around annotation values to find whitespaces</xsl:variable>
	<xsl:variable name="TOOLTIP_EXPAND">Expands/Collapses annotation values</xsl:variable>
    
    <!-- general descriptions -->
    <xsl:variable name="DESC_SHORT">short description of myCorpus</xsl:variable>
    <xsl:variable name="DESC_CORPUS">Here you can enter a description of your corpus.</xsl:variable>
    <xsl:variable name="DEFAULT_ANNOTATOR_M">John Doe</xsl:variable>
    <xsl:variable name="DEFAULT_EMAIL_M">john-doe@sample.com</xsl:variable>
    <xsl:variable name="DEFAULT_ANNOTATOR_W">Jane Doe</xsl:variable>
    <xsl:variable name="DEFAULT_EMAIL_W">jane-doe@sample.com</xsl:variable>
    <xsl:variable name="LICENSE">No license is given for this corpus.</xsl:variable>
    
    <!-- description texts for sections -->
    <xsl:variable name="DESC_STRUCTURAL_INFO">Structural data are those, which were necessary to create the Salt model. Since Salt is a graph-based model, all model elements are either nodes or relations between them. Salt contains a set of subtypes of nodes and relations. Subtypes of nodes are: SToken, STextualDS (primary data), SSpan, SStructure and some more. Subtypes of relations are: SSpanningRelation, SDominanceRelation, SPointingRelation and some more. This section gives an overview of the frequency (Count) for each of those elements (Name) used in this corpus or document.</xsl:variable>
    <xsl:variable name="DESC_META_DATA">Meta data of a document or a corpus provide information about its origin and the annotation process. Meta data for instance can give information on where the primary data came from, who annotated it, which tools have been used and so on. Thereby the row 'Name' shows the respective categories like 'writer' and the row 'Values' contains a list of the represented values like 'Goethe', 'Schiller', etc.</xsl:variable>
    <xsl:variable name="DESC_ANNO_DESC">This section displays all annotations contained in this corpus or document</xsl:variable>
    <xsl:variable name="DESC_ANNO_DESC_1">which does not belong to any layer. Annotations being contained in layers are displayed below. Annotations in Salt are attribute-value-pairs. This table contains the frequencies of all annotation names (Name) and annotation values (Values).</xsl:variable>
    <xsl:variable name="DESC_ANNO_DESC_2">. Annotations in Salt are attribute-value-pairs. This table contains the frequencies of all annotation names (Name) and annotation values (Values).</xsl:variable>
    <xsl:variable name="DESC_SLAYER">Annotations in Salt are attribute-value-pairs. This table contains the frequencies of all annotation names (Name) and annotation values (Values).</xsl:variable>
    
    <!-- buid html sceleton-->
    <xsl:template match="sCorpusInfo|sDocumentInfo">
        <html>
            <head>
                <META http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
                
                <link href="./css/jquery-ui.css" rel="stylesheet"/>
                <script src="./js/jquery.js"></script>
                <script src="./js/jquery-ui.js"></script>
            </head>
            <body>
                <div align="right">
                   <a class="help" onclick="loadHelpPage();"><i class="fa fa-question-circle"/> Help</a>
                </div>
                <!-- get corpus name-->
                <h2 id="title">
                    <xsl:value-of select="@sName"/>
                </h2>
                <div>
                    <!-- get meta info table -->
                    <xsl:apply-templates select="metaDataInfo"/>
                </div>
                
                <!-- get annotation info table -->
                <xsl:if test="sAnnotationInfo">
                <xsl:call-template name="annoTable"/>
                </xsl:if>

                <!-- set meta data info as json input -->
                <xsl:if test="exists(//sAnnotationInfo[count(.//sValue) &gt; $minNumOfAnnos])">
                    <xsl:result-document href="{$jsonOutputName}" format="json">
                            <xsl:call-template name="json"/>
                    </xsl:result-document>
                </xsl:if>
                    
                <!-- create main.html if current file is main corpus -->
                <xsl:if test="$isMainCorpus"> 
                    <xsl:result-document href="main.html" format="main">
                        <xsl:call-template name="main"/>
                    </xsl:result-document>
                </xsl:if>
                
                <!-- create customization.json if current file is main corpus -->
                <xsl:if test="$isMainCorpus"> 
                    <xsl:result-document href="customization.json" format="customization">
                        <xsl:call-template name="customization"/>
                    </xsl:result-document>
                </xsl:if>
                
                <!-- get layer info table  -->
                <xsl:apply-templates select="sLayerInfo">
                    <xsl:sort select="@sName" lang="de"/>
                </xsl:apply-templates>
                
                <!-- get structural info table -->
                <xsl:apply-templates select="structuralInfo"/>
                
                <!-- insert javascript code -->
                <script>
                  	 addTooltips_MetaData();
                  	 addTooltips_AnnotationNames();
                  	 styleToolTips();
                  	 addDescs();
                </script>
            </body>
        </html>
    </xsl:template>
    
    <!-- build structural info table -->
    <xsl:template name="structInfo" match="structuralInfo">
        <xsl:if test="not(empty(child::node()))">
        <h3>Structural Info</h3>
        <hr/>
        <!-- paragraph for description -->
        <p id="structInfoDescription">
        </p>
        <!-- create table structure -->
        <table class="data-structuralInfo">
            <thead>
                <th>Name</th>
                <th>Frequency</th>
            </thead>
            <tbody>
                <!-- set all structural entries -->
                <xsl:apply-templates select="entry" mode="structEntry">
                    <xsl:sort select="@key" lang="de"/>
                </xsl:apply-templates>
            </tbody>
        </table>
        </xsl:if>
    </xsl:template>

    <!-- get all structural entries of the corpus -->
    <xsl:template match="entry" mode="structEntry">
        <!-- get position of the entry and set class name for background colors -->
        <xsl:variable name="entry" select="position()"/>
        <!-- seperate every second value to differ in background color in every second row -->
        <xsl:choose>
            <xsl:when test="$entry mod 2=1">
                <tr class="odd">
                    <td class="entry-key">
                        <!-- include tooltips for structural infos -->
                        <span class="tooltip">
                            <xsl:attribute name="title"><xsl:call-template name="structTooltips"/></xsl:attribute>
                            <xsl:value-of select="@key"/>
                                <i class="fa fa-info-circle icon"/>
                        </span>
                    </td>
                    <td>
                        <xsl:value-of select="text()"/>
                    </td>
                </tr>
            </xsl:when>
            <xsl:when test="$entry mod 2=0">
                <tr class="even">
                    <td class="entry-key">
                        <span class="tooltip">
                            <xsl:attribute name="title"><xsl:call-template name="structTooltips"/></xsl:attribute>
                            <xsl:value-of select="@key"/>
                                <i class="fa fa-info-circle icon"/>
                        </span>
                    </td>
                    <td>
                        <xsl:value-of select="text()"/>
                    </td>
                </tr>
            </xsl:when>
        </xsl:choose>
    </xsl:template>

<!-- build meta data table -->
    <xsl:template match="metaDataInfo">
        <xsl:if test="not(empty(child::node()))">
            <br/>
        <h3>Meta Data</h3>
        <hr/>
        <!-- paragraph for description -->
        <p id="metaDataDescription">
        </p>
        <table>
            <thead>
                <th>Name</th>
                <th>Values</th>
            </thead>
            <tbody>
                <!-- set metadata entries -->
                <xsl:apply-templates select="entry" mode="metaEntry">
                    <xsl:sort select="@key" lang="de"/>
                </xsl:apply-templates>
            </tbody>
        </table>
        </xsl:if>
    </xsl:template>

    <!-- get first 5 metadata entries -->
    <xsl:template match="entry" mode="metaEntry">
        <!-- get position of the entry and set class name for background colors -->
        <xsl:variable name="entry" select="position()"/>
        <xsl:choose>
            <xsl:when test="($entry mod 2=1)">
                <tr class="odd">
                    <td class="entry-key">
                        <span class="metadata-name">
                            <xsl:value-of select="@key"/>
                        </span>
                    </td>
                    <td>
                        <xsl:value-of select="text()"/>
                    </td>
                </tr>
            </xsl:when>
            <xsl:when test="($entry mod 2=0)">
                <tr class="even">
                    <td class="entry-key">
                        <span class="metadata-name">
                            <xsl:value-of select="@key"/>
                        </span>
                    </td>
                    <td>
                        <xsl:value-of select="text()"/>
                    </td>
                </tr>
            </xsl:when>
        </xsl:choose>
    </xsl:template>

<!-- build annotation table -->
    <xsl:template name="annoTable">
        <xsl:if test="not(empty(sAnnotationInfo/child::node()))">
            <br/>
        <div>
        <h3>Annotations</h3>
        <hr/>
            <!-- paragraph for description -->
            <p id="annoDescription">
            </p>
        <table class="data-table">
            <thead>
                <th>Name</th>
                <th>Values</th>
            </thead>
            <tbody>
                <!-- set metadata entries -->
                <xsl:apply-templates select="sAnnotationInfo" mode="annoTable">
                    <xsl:sort select="@sName" lang="de"/>
                </xsl:apply-templates>
            </tbody>
        </table>
        </div>
        </xsl:if>
    </xsl:template>

<!-- get annotation values -->
    <xsl:template match="sAnnotationInfo" mode="annoTable">
        <!-- get position of the entry and set class name for background colors -->
        <xsl:variable name="sName" select="position()"/>
        <xsl:choose>
            <xsl:when test="($sName mod 2=1)">
                <tr class="odd">
                    <xsl:call-template name="annoContent"/>
                </tr>
            </xsl:when>
            <xsl:when test="($sName mod 2=0)">
                <tr class="even">
                   <xsl:call-template name="annoContent"/>
                </tr>
            </xsl:when>
        </xsl:choose>
    </xsl:template>

    <!-- get first 5 occurences -->
    <xsl:template match="sValue">
        <xsl:choose>
            <xsl:when test="position() &lt; $minNumOfAnnos+1">
                <span class="svalue">
                    <!-- include link to annis -->
                    <span class="svalue-text" onmouseover="clickifyMe($(this));" onmouseout="$(this).removeClass(CLASS_CLICKIFY);$(this).addClass(CLASS_DECLICKIFY);">
                        <xsl:attribute name="onclick">goANNIS('<xsl:value-of select="./parent::sAnnotationInfo/@sName"></xsl:value-of>', this.innerHTML);</xsl:attribute><xsl:value-of select="text()"/>
                    </span>
                    <span class="anno-value-count">
                        <xsl:value-of select="@occurrence"/>
                    </span>
                </span>
            </xsl:when>
        </xsl:choose>
    </xsl:template>
    
    <!-- create table content for annotations -->
    <xsl:template name="annoContent">
        <td class="entry-key">
            <span class="sannotationinfo">
                <span class="anno-sname" onmouseover="clickifyMe($(this));"
                    onmouseout="$(this).removeClass(CLASS_CLICKIFY);$(this).addClass(CLASS_DECLICKIFY);"
                    onclick="goANNIS(this.innerHTML);">
                    <xsl:value-of select="normalize-unicode(normalize-space(replace(replace(@sName, '\\','\\\\'), '&quot;', '\\&quot;')))"/>
                </span>
                <span class="anno-count">
                    <xsl:value-of select="@occurrence"/>
                </span>
            </span>
            <i class="fa fa-download btn-download-csv icon tooltip" title="{$TOOLTIP_DOWNLOAD}"/>
            <i class="fa fa-square-o btn-toggle-box icon tooltip" title="{$TOOLTIP_BOXES}"/>
            <!-- if current value is not one of the first annotation values, enable loading of values from the json file -->
            <xsl:choose>
                <xsl:when test="count(sValue) &lt; $minNumOfAnnos+1"></xsl:when>
                <xsl:otherwise>
                    <i class="fa fa-expand icon tooltip" title="{$TOOLTIP_EXPAND}">
                            <xsl:attribute name="id">
                                <xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if>
                                <xsl:value-of select="normalize-unicode(normalize-space(replace(replace(@sName, '\\','\\\\'), '&quot;', '\\&quot;')))"/><xsl:text>_btn</xsl:text>
                            </xsl:attribute>
                            <xsl:attribute name="onClick">
                                <xsl:text>loadAndExpandAnnoValues('</xsl:text><xsl:value-of select="$jsonOutputPath"/><xsl:text>','</xsl:text>
                                <xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="normalize-unicode(normalize-space(replace(replace(@sName, '\\','\\\\'), '&quot;', '\\&quot;')))"/><xsl:text>')</xsl:text>
                            </xsl:attribute>
                    </i>
                </xsl:otherwise>
            </xsl:choose>
        </td>
        <td>
            <!-- Print annotation values and their occurrence -->
            <xsl:attribute name="id">
                <xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="normalize-unicode(normalize-space(replace(replace(@sName, '\\','\\\\'), '&quot;', '\\&quot;')))"/>
                <xsl:text>_values</xsl:text>
            </xsl:attribute>
            <xsl:apply-templates select="sValue">
                <xsl:sort select="text()" lang="de"/>
            </xsl:apply-templates>
        </td>
    </xsl:template>
    
    <!-- create entries for layer annotation -->
    <xsl:template match="sLayerInfo" mode="layerJson">
        <xsl:apply-templates select="sAnnotationInfo" mode="annoJson">
            <xsl:sort select="@sName" lang="de"/>
        </xsl:apply-templates>
    </xsl:template>

    <!-- create the json file for annotations -->
    <xsl:template match="sAnnotationInfo" mode="annoJson">        <xsl:choose>
        <!-- print all annotations independed of the amount of annotation values into the json file -->
        <xsl:when test="$createJsonForAllAnnos">
            <xsl:variable name="curName"><xsl:value-of select="@sName"/></xsl:variable>
            <!-- print sLayer name before annotation layer if one exists -->
            "<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="@sName"/>": [
            <xsl:apply-templates select="sValue" mode="ValueJson">
                <xsl:sort select="text()"  lang="de"/>
            </xsl:apply-templates>
            <xsl:choose>
                <!-- check for following sLayer -->
                <xsl:when test="exists(following::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0]) or exists(preceding::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0])">],
                </xsl:when>
                <xsl:otherwise>]
                </xsl:otherwise>
            </xsl:choose>
        </xsl:when>
    <xsl:otherwise>
        <!-- only print those annotations into the json file, that include more than 5 annotation values -->
        <xsl:if test="count(.//sValue) > $minNumOfAnnos">"<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if>
            <xsl:value-of select="normalize-unicode(normalize-space(replace(replace(@sName, '\\','\\\\'), '&quot;', '\\&quot;')))"/>": [
            <xsl:apply-templates select="sValue" mode="ValueJson">
                <xsl:sort select="text()" lang="de"/>
            </xsl:apply-templates>
            <xsl:variable name="curName"><xsl:value-of select="@sName"/></xsl:variable>
            <xsl:choose>
                <!-- print sLayer  -->
            <xsl:when test="parent::sLayerInfo">
                <xsl:variable name="curSLayer"><xsl:value-of select="parent::sLayerInfo/@sName"/></xsl:variable>
            <xsl:choose>
                <!-- check for annotations that are following the current one, lexically -->
                <xsl:when test="exists(following-sibling::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0 and count(.//sValue) &gt; $minNumOfAnnos]) or exists(preceding-sibling::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0 and count(.//sValue) &gt; $minNumOfAnnos]) or exists(following::sLayerInfo[compare(upper-case($curSLayer), upper-case(@sName)) &lt; 0 and exists(./sAnnotationInfo[count(.//sValue) &gt; $minNumOfAnnos])]) or exists(preceding::sLayerInfo[compare(upper-case($curSLayer), upper-case(@sName)) &lt; 0 and exists(./sAnnotationInfo[count(.//sValue) &gt; $minNumOfAnnos])])">],
                </xsl:when>
                <xsl:otherwise>]</xsl:otherwise>
            </xsl:choose></xsl:when>
                <xsl:otherwise>
                    <xsl:choose>
                        <xsl:when test="exists(following-sibling::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0 and count(.//sValue) &gt; $minNumOfAnnos]) or exists(preceding-sibling::sAnnotationInfo[compare(upper-case($curName), upper-case(@sName)) &lt; 0 and count(.//sValue) &gt; $minNumOfAnnos]) or exists(//sLayerInfo[count(//sAnnotationInfo//sValue) &gt; $minNumOfAnnos])">],
                        </xsl:when>
                        <xsl:otherwise>]
                        </xsl:otherwise>
                    </xsl:choose>
                </xsl:otherwise>
            </xsl:choose> </xsl:if>
         </xsl:otherwise></xsl:choose>
         </xsl:template>

<!-- get annotation values and normalize strings -->
    <xsl:template match="sValue" mode="ValueJson">{"value":"<xsl:value-of select="normalize-unicode(normalize-space(replace(replace(text(), '\\','\\\\'), '&quot;', '\\&quot;')))"/>", "occurrence": "<xsl:value-of select="@occurrence"/>
        <xsl:choose>
            <xsl:when test="position()!=last()">"},
        </xsl:when>
        <xsl:otherwise>"}
        </xsl:otherwise>
    </xsl:choose>
    </xsl:template>
    
    <!-- build json file for annotations -->
    <xsl:template name="json">{
        <xsl:apply-templates select="sAnnotationInfo" mode="annoJson">
                <xsl:sort select="@sName" lang="de"/>
        </xsl:apply-templates>
        <xsl:apply-templates select="sLayerInfo" mode="layerJson">
            <xsl:sort select="@sName" lang="de"/>
        </xsl:apply-templates>
        }
    </xsl:template>
    
    <!-- build sLayer table for each sLayer -->
    <xsl:template match="sLayerInfo">
        <xsl:if test="not(empty(child::node()))">
            <br/>
        <div>
            <h3><xsl:value-of select="@sName"/></h3>
            <hr/>
            <!-- paragraph for description -->
            <p class="layer-desc">
                <xsl:attribute name="id"><xsl:value-of select="@sName"/>_desc</xsl:attribute>
            </p>
            <table class="data-table">
                <thead>
                    <th>Name</th>
                    <th>Values</th>
                </thead>
                <tbody>
                    <!-- set metadata entries -->
                    <xsl:apply-templates select="sAnnotationInfo" mode="annoTable">
                        <xsl:sort select="@sName" lang="de"/>
                    </xsl:apply-templates>
                </tbody>
            </table>
        </div>
        </xsl:if>
    </xsl:template>
    
    <!-- create main page with detailed corpus description if given (customization.json) -->
    <xsl:template name="main">
        <html>
            <head>
                <title>
                    Main
                </title>
            </head>
            
            <div align="right">
               <a class="help" onclick="loadHelpPage();"><i class="fa fa-question-circle"/> Help</a>
            </div>
            <body>
                <h2 id="corpusTitle">
                    <xsl:value-of select="$currentFile"/>
                </h2>
                <hr/>
                <article id="corpusDescription">
                </article>
                <article id="annotators">
	        	<h3>Annotators</h3>
                </article>
                <article id="license">
			<h3>License</h3>
              	</article>
            </body>
        </html>
    </xsl:template>
    
    
    <!-- create customization file with information like corpus name, corpus description etc. -->
    <xsl:template name="customization">{
        "corpusName" : "<xsl:value-of select="$corpusname"/>",
        "shortDescription" : "<xsl:value-of select="$DESC_SHORT"></xsl:value-of>",
        "description" : "<xsl:value-of select="$DESC_CORPUS"></xsl:value-of>",
        "annotators" : [ 
            {"name" : "<xsl:value-of select="$DEFAULT_ANNOTATOR_M"/>", "eMail" : "<xsl:value-of select="$DEFAULT_EMAIL_M"></xsl:value-of>"}, 
        {"name" : "<xsl:value-of select="$DEFAULT_ANNOTATOR_W"/>", "eMail" : "<xsl:value-of select="$DEFAULT_EMAIL_W"></xsl:value-of>"}
        ],
        "license": "<xsl:value-of select="$LICENSE"/>",
        <xsl:if test="not(empty(//metaDataInfo//entry))">
        "tooltips_metadata" : [<xsl:apply-templates mode="metaTooltips" select="metaDataInfo"><xsl:sort select="@key"></xsl:sort></xsl:apply-templates>
        ],
        </xsl:if>
        <xsl:if test="not(empty(//sAnnotationInfo))">
            "tooltips_annonames" : [
            <xsl:call-template name="loop"><xsl:with-param name="index">0</xsl:with-param>
                <xsl:with-param name="max"><xsl:value-of select="$NumOfTooltips"/></xsl:with-param></xsl:call-template>],
        </xsl:if>
         <!--deprecated json-info:-->
<!--        "annisLink" : "https://korpling.german.hu-berlin.de/annis3/",-->
         "structInfoDesc" : "<xsl:value-of select="$DESC_STRUCTURAL_INFO"/>",
        "metaDataDesc" : "<xsl:value-of select="$DESC_META_DATA"/>"<xsl:if test="exists(//sAnnotationInfo)">,
            "annoDesc" : "<xsl:value-of select="$DESC_ANNO_DESC"/></xsl:if><xsl:choose>
            <xsl:when test="exists(//sLayerInfo)"> <xsl:value-of select="$DESC_ANNO_DESC_1"/>"</xsl:when>
            <xsl:otherwise><xsl:value-of select="$DESC_ANNO_DESC_2"/>"</xsl:otherwise>
        </xsl:choose><xsl:choose><xsl:when test="exists(//sLayerInfo)">,
        "layerDesc" : [
        <xsl:apply-templates mode="sLayerDesc" select="sLayerInfo">
            <xsl:sort select="@sName" lang="de"/>
        </xsl:apply-templates>]</xsl:when></xsl:choose>
        }
    </xsl:template>
    
    <!-- create a loop to iterate $NumOfTooltips-times -->
    <xsl:template name="loop">
        <xsl:param name="index"></xsl:param>
        <xsl:param name="max"></xsl:param>
        <xsl:if test="not($index + count(preceding::sAnnotationInfo) &gt; $max)">
            <xsl:apply-templates mode="annoTooltips" select="sAnnotationInfo[position() = $index]">
                <xsl:with-param name="index"><xsl:value-of select="$index"/></xsl:with-param>
                <xsl:with-param name="max"><xsl:value-of select="$max"/></xsl:with-param>
                <xsl:sort select="@sName" lang="de"/>
            </xsl:apply-templates>
            <xsl:if test="not(exists(sAnnotationInfo[not(parent::sLayerInfo)])) and count(sAnnotationInfo[not(parent::sLayerInfo)]) &lt; $NumOfTooltips">
                <xsl:apply-templates mode="annoTooltips" select="sLayerInfo/sAnnotationInfo[position() = $index + count(sAnnotationInfo[not(parent::sLayerInfo)])]">
                    <xsl:with-param name="index"><xsl:value-of select="$index"/></xsl:with-param>
                    <xsl:with-param name="max"><xsl:value-of select="$max"/></xsl:with-param>
                    <xsl:sort select="@sName" lang="de"/>
                </xsl:apply-templates>
            </xsl:if>
            <xsl:call-template name="loop">
                <xsl:with-param name="index"><xsl:value-of select="$index + 1"/></xsl:with-param>
                <xsl:with-param name="max"><xsl:value-of select="$max"/></xsl:with-param>
            </xsl:call-template>
            <xsl:if test="exists(sAnnotationInfo[not(parent::sLayerInfo)]) and count(preceding::sAnnotationInfo[not(parent::sLayerInfo)])  &lt; $NumOfTooltips">
                <xsl:apply-templates mode="annoTooltips" select="sLayerInfo/sAnnotationInfo[position() = $index + count(sAnnotationInfo[not(parent::sLayerInfo)]) - count(preceding::sAnnotationInfo[parent::sLayerInfo])]">
                    <xsl:with-param name="index"><xsl:value-of select="$index + count(sAnnotationInfo[not(parent::sLayerInfo)])"/></xsl:with-param>
                    <xsl:with-param name="max"><xsl:value-of select="$max"/></xsl:with-param>
                    <xsl:sort select="@sName" lang="de"/>
                </xsl:apply-templates>
            </xsl:if>
        </xsl:if>
    </xsl:template>
    
    
    <!-- create first "NumOfTooltips" (3) tooltips for meta data -->
    <xsl:template match="entry" mode="metaEntryTooltip">
        <xsl:choose><xsl:when test="(position() &lt; $NumOfTooltips) and position() != last()">
            {"name": "<xsl:value-of select="@key"/>", "tooltip": ""},</xsl:when>
            <xsl:when test="(position() = $NumOfTooltips) or ((count(following-sibling::entry) + count(preceding-sibling::entry) &lt; $NumOfTooltips) and position() = last())">
                {"name": "<xsl:value-of select="@key"/>", "tooltip": ""}</xsl:when></xsl:choose>
    </xsl:template>
    
    <!-- set tooltips for the first "NumOfTooltips" annotations  -->
    <xsl:template match="sAnnotationInfo" mode="annoTooltips">
       <xsl:param name="index"/> 
        <xsl:param name="max"/>
        <xsl:if test="$index &lt; $max and (exists(following::sAnnotationInfo) or exists(//sLayerInfo))">{"name": "<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="@sName"/>", "tooltip": ""},
            </xsl:if>
        <xsl:if test="$index = $max or (count(//sAnnotationInfo[not(parent::sLayerInfo)]) &lt; $NumOfTooltips and not(exists(following::sAnnotationInfo)) and not(exists(//sLayerInfo)))">{"name": "<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="@sName"/>", "tooltip": ""}
        </xsl:if>
    </xsl:template>
    
    <!-- create tooltips for slayer -->
    <xsl:template mode="annoTooltips" match="sLayerInfo/sAnnotationInfo">
        <xsl:param name="index"/> 
        <xsl:param name="max"></xsl:param>
        <xsl:if test="$index &lt; $max and (not(count(//sAnnotationInfo[not(parent::sLayerInfo)]) + count(//sAnnotationInfo[parent::sLayerInfo]) &lt; $NumOfTooltips))">{"name": "<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="@sName"/>", "tooltip": ""},
        </xsl:if>
        <xsl:if test="$index = $max or (count(//sAnnotationInfo[parent::sLayerInfo]) + count(//sAnnotationInfo[not(parent::sLayerInfo)]) &lt; $NumOfTooltips and position() = last())">{"name": "<xsl:if test="parent::sLayerInfo"><xsl:value-of select="../@sName"/>:</xsl:if><xsl:value-of select="@sName"/>", "tooltip": ""}
        </xsl:if>
    </xsl:template>
    
    <!-- build default layer descriptions for each layer -->
    <xsl:template mode="sLayerDesc" match="sLayerInfo">
        <xsl:variable name="curName"><xsl:value-of select="@sName"/></xsl:variable>
        {"name" : "<xsl:value-of select="@sName"/>_desc" , "desc" : "These are the annotations for the <xsl:value-of select="@sName"/> layer. <xsl:value-of select="$DESC_SLAYER"/>"
        }<xsl:if test="exists(following-sibling::sLayerInfo[compare($curName, @sName) &lt; 0]) or exists(preceding-sibling::sLayerInfo[compare(@sName,current()/@sName)&gt;0])">,</xsl:if>
    </xsl:template>
    
    <!-- set tooltips for meta data entries -->
    <xsl:template match="metaDataInfo" mode="metaTooltips">
        <xsl:apply-templates mode="metaEntryTooltip" select="entry">
            <xsl:sort select="@key"/>
        </xsl:apply-templates>
    </xsl:template>
 
    <!-- choose matching tooltip for structual info -->
    <xsl:template name="structTooltips">
        <xsl:choose>
            <xsl:when test="@key = 'SNode'"><xsl:value-of select="$SNode"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SRelation'"><xsl:value-of select="$SRelation"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SSpan'"><xsl:value-of select="$SSpan"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SSpanningRelation'"><xsl:value-of select="$SSpanningRelation"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'STextualDS'"><xsl:value-of select="$STextualDS"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'STimeline'"><xsl:value-of select="$STimeline"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SToken'"><xsl:value-of select="$SToken"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SPointingRelation'"><xsl:value-of select="$SPointingRelation"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'STextualRelation'"><xsl:value-of select="$STextualRelation"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SStructure'"><xsl:value-of select="$SStructure"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'SDominanceRelation'"><xsl:value-of select="$SDominanceRelation"/></xsl:when>
        </xsl:choose>
        <xsl:choose>
            <xsl:when test="@key = 'STimelineRelation'"><xsl:value-of select="$STimelineRelation"/></xsl:when>
        </xsl:choose>
    </xsl:template>

</xsl:stylesheet>
