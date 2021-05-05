/*global $:false */
'use strict';
/** call function main, when document was loaded entirely */
$(document).ready(main);

/**
 * This method is called when complete document was loaded. This makes this function a main method.
 * */
function main() {
	/** Adds CSV download functionality to button or icon */
    $("#content").on("click", ".btn-download-csv", function(event) {
        var data = $(this).parent().next().children('.svalue');
        downloadText(convertToCSV(data), CSV_MIME_TYPE);
    });

    /** Creation of boxes around annotation values*/
    $("#content").on("click", ".btn-toggle-box", toggleBox);

    // loads customization file into global variables
    loadCustomization();

    // Load content for main page
    loadMainPage();
};

/*******************************************************************************
 * CSV download vor annotation values
 ******************************************************************************/
var CSV_SEPARATOR = ',';
var CSV_DOUBLEQUOTE = '"';
var CSV_LINEBREAK = '\r\n';
var CSV_MIME_TYPE = 'text/csv';
var CLASS_OCCURRENCE= '.anno-value-count';
var INTERPRET_AS_HTML = false;

/**
 * loads text as data uri
 */
function downloadText(text, mime) {
    window.location.href = 'data:' + mime + ';charset=UTF-8,' + encodeURIComponent(text);
}

/**
 * escapes double-quotes https://tools.ietf.org/html/rfc4180#section-2
 * Section 7
 */
function escapeDQuote(string) {
    return string.replace(/"/g, '""');
}

/**
 * Converts the given array of svalue-data items into an csv text. Values are
 * quoted because there could be line breaks see
 * https://tools.ietf.org/html/rfc4180
 */
function convertToCSV(svalues) {
    var text = '';
    $(svalues).each(
        function() {
            var valuename = escapeDQuote($(this).children(
                '.svalue-text').text());
            var valuecount = $(this).children(CLASS_OCCURRENCE)
                .text();
            text += CSV_DOUBLEQUOTE + valuename + CSV_DOUBLEQUOTE;
            text += CSV_SEPARATOR;
            text += CSV_DOUBLEQUOTE + valuecount + CSV_DOUBLEQUOTE;
            text += CSV_LINEBREAK;
        });
    text += CSV_LINEBREAK;
    return text;
}

/***************************************************************************
 * Boxes for annotation values
 **************************************************************************/
function toggleBox(event) {
    var values = $(this).parent().next().children().children('.svalue-text');
    $(values).toggleClass('boxed');
}

/*******************************************************************************
 * Load customization file (customization.json) into variables
 ******************************************************************************/
/** customization file containing user defined vaules to adapt web site */
var FILE_CUSTOMIZATION = "./customization.json";
/** Contains the name of the root corpus */
var corpusName = "";
/** Contains a short description of the corpus if given */
var shortDescription = "";
/** Contains a long description of the corpus if given */
var description = "";
/** Contains a description of "structuralInfo" if given */
var structInfoDesc = "";
/** Contains a description of "metaDataInfo" if given */
var metaDataDesc = "";
/** Contains a description of "sAnnotationInfo" if given */
var annoDesc = "";
/** A table containing description of layer annotations */
var layerDesc = [];
/** Contains an array of author names*/
var annotators = [];
/** Contains the license of the corpus**/
var license = "";
/** Link to ANNIS instance **/
var annisLink = null;
/** A table containing tooltips for metadata **/
var tooltips_metadata = null;
/** A table containing tooltips for annotation  **/
var tooltips_annonames = null;
/** A table containing tooltips for structure part  **/
var tooltips_structuralInfo=null;

/** Defines an object of type Author having a name and aemail address**/
function Author(name, eMail) {
    this.name = name;
    this.eMail = eMail;
}

function Layer(name, desc) {
    this.name = name;
    this.desc = desc;
}


/** loads customization file and files variables **/
function loadCustomization() {
    //set the MIME type to json, otherwise firefoy produces a warning
    $.ajaxSetup({
        beforeSend: function(xhr) {
            if (xhr.overrideMimeType) {
                xhr.overrideMimeType("application/json");
            }
        }
    });
    /** load customization file */
    $.getJSON(FILE_CUSTOMIZATION, function(json) {
	corpusName = json.corpusName;        
	shortDescription = json.shortDescription;
        description = json.description;
        license = json.license;
        structInfoDesc = json.structInfoDesc;
        metaDataDesc = json.metaDataDesc;
        annoDesc = json.annoDesc;
        
        if (INTERPRET_AS_HTML){
        	$("#project_tagline").append(shortDescription);
        } else {
            $("#project_tagline").text(shortDescription);
        }
        
       	if (typeof json.annotators!== "undefined"){
		for (var i = 0; i < json.annotators.length; i++) {
		    annotators[annotators.length] = new Author(json.annotators[i].name, json.annotators[i].eMail);
		}
        }
	if (typeof json.layerDesc!== "undefined"){
		for (var i = 0; i < json.layerDesc.length; i++) {
			layerDesc[layerDesc.length] = new Layer(json.layerDesc[i].name, json.layerDesc[i].desc);
		}
	}
        //load annis links
        annisLink = json.annisLink;
        if (annisLink != null) {
            $("#search_me").css("visibility", "visible");
        }
        // load tooltips for meta data
	if (json.tooltips_metadata!= null){
		for (var i = 0; i < json.tooltips_metadata.length; i++) {
			if (tooltips_metadata== null){
				tooltips_metadata = new Object();
			}
			if (json.tooltips_metadata[i].tooltip){
				tooltips_metadata[json.tooltips_metadata[i].name] = json.tooltips_metadata[i].tooltip;
			}
		}
	}
	
    if (tooltips_metadata== null){
		console.debug("No tooltips for metadata found in file '"+FILE_CUSTOMIZATION+"'. ");
	}

        
		
		// load tooltips for annotation names
		if (json.tooltips_annonames!= null){
			for (var i = 0; i < json.tooltips_annonames.length; i++) {
				if (tooltips_annonames== null){
					tooltips_annonames = new Object();
				}
				if (json.tooltips_annonames[i].tooltip){
					tooltips_annonames[json.tooltips_annonames[i].name] = json.tooltips_annonames[i].tooltip;
				}
			}
		}
        if (tooltips_annonames== null){
			console.debug("No tooltips for annotation names found in file '"+FILE_CUSTOMIZATION+"'. ");
		}
		
		 if (json.tooltips_structuralInfo!= null){
			for (var i = 0; i < json.tooltips_structuralInfo.length; i++) {
				if (tooltips_structuralInfo== null){
					tooltips_structuralInfo = new Object();
				}
				tooltips_structuralInfo[json.tooltips_structuralInfo[i].name] = json.tooltips_structuralInfo[i].tooltip;
			}
		}
        if (tooltips_structuralInfo== null){
			console.debug("No tooltips for structural info found in file '"+FILE_CUSTOMIZATION+"'. ");
		}
		
    });
}

/*******************************************************************************
 * Collapse/expand the annotation values corresponding to an annotation name,
 * when there are more annotations as predefined treshhold.
 ******************************************************************************/
var NUM_OF_SET_VALUES = 5;
var SYMBOL_COLLAPSE = "fa fa-compress";
var SYMBOL_EXPAND = "fa fa-expand";
var annoTable = null;
// object to store what is the current loaded file
var currentFile= null;
/**
 * Loads the anno map from passed file if necessary and expands
 * the cell corresponding to passed annoName
 */
function loadAndExpandAnnoValues(file, annoName) {
    //if annoTable wasn't load with passed file, load it now
    if (currentFile!= file){
		currentFile= file;
        if (file != null) {
            //set the MIME type to json, otherwise firefoy produces a warning
            $.ajaxSetup({
                beforeSend: function(xhr) {
                    if (xhr.overrideMimeType) {
                        xhr.overrideMimeType("application/json");
                    }
                }
            });
	    $.getJSON(file, function(json) {
		annoTable = json;
                expandAnnoValues(annoName);
            });
        } else {
            console.error("Cannot load annotation map file, since the passed file was empty.");
        }
    } else {
        expandAnnoValues(annoName);
    }
}

/**
 * Expands the annotation values for the cell corresponding to
 * passed annoName.
 **/
function expandAnnoValues(annoName) {
        var slot= annoTable[annoName];
	if (typeof slot=== "undefined"){
		console.warn("No entry in json found for anno name '"+annoName+"'. ");
	}
	var id= "#"+annoName.replace(":", "\\:");
	
	var $td = $(id+ "_values");
        var $span = $td.children().eq(0);

        for (var i = NUM_OF_SET_VALUES; i < slot.length; i++) {
            var $newSpan = $span.clone();
            $newSpan.children().eq(0).text(slot[i].value);
            $newSpan.children().eq(1).text(slot[i].occurrence);
            $td.append($newSpan);
        }

        var $btn = $(id + "_btn");
        $btn.removeClass(SYMBOL_EXPAND);
        $btn.addClass(SYMBOL_COLLAPSE);
        $btn.unbind('click');
        $btn.attr("onclick", "collapseValues('" + annoName + "')");
    }
/**
 * Collapses the annotation values for the cell corresponding to
 * passed annoName.
 * For better performance, first all childs of the concerning <td> element 
 * are removed and second the first NUM_OF_SET_VALUES are added again. This 
 * is much faster than removing all elements after element NUM_OF_SET_VALUES.
 **/
function collapseValues(annoName) {
	var slot= annoTable[annoName];
	if (typeof slot=== "undefined"){
		console.warn("No entry in json found for anno name '"+annoName+"'. ");
	}	
	var id= "#"+annoName.replace(":", "\\:");
        var $td = $(id + "_values");
	var $span = $td.children().eq(0).clone();
	$td.empty();
	for (var i = 0; i < NUM_OF_SET_VALUES; i++) {
	    var $newSpan = $span.clone();
            $newSpan.children().eq(0).text(slot[i].value);
            $newSpan.children().eq(1).text(slot[i].occurrence);
            $td.append($newSpan);
        }
        var $btn = $(id + "_btn");
        $btn.removeClass(SYMBOL_COLLAPSE);
        $btn.addClass(SYMBOL_EXPAND);
        $btn.unbind('click');
        $btn.attr("onclick", "expandAnnoValues('" + annoName + "')");
    }
/*******************************************************************************
 * Add the jQuery Tooltip styling mechanism to tooltip elements and style them
 * see: http://jqueryui.com/tooltip/
 ******************************************************************************/
function styleToolTips() {
    $('.tooltip').tooltip({
        show: "true",
        close: function(event, ui) {
            ui.tooltip.hover(function() {
                    $(this).stop(true).fadeTo(10, 1);
                },
                function() {
                    $(this).fadeOut('10', function() {
                        $(this).remove();
                    });
                });
        }
    });
};




/*******************************************************************************
 * Adding tooltips for metadata and annotation names
 ******************************************************************************/
var CLASS_METADATA="metadata-name";
var CLASS_ANNO_NAMES="anno-sname";
var CLASS_TOOLTIP="tooltip";
var CLASS_LAYER_NAMES="layer-desc";
/** 
 * Adds tootlips to all elements having the class CLASS_METADATA.
 **/
function addTooltips_MetaData() {
    if (tooltips_metadata!= null){
		//find all elements of class CLASS_METADATA
		var metaElements = document.getElementsByClassName(CLASS_METADATA);
		for (var i= 0; i < metaElements.length; i++){
			var tooltip= tooltips_metadata[metaElements[i].innerHTML];
			if (	(tooltip!= null)&&
					(tooltip!= "")){
				metaElements[i].title=tooltip;
				$(metaElements[i]).addClass(CLASS_TOOLTIP);
			}
		}
	}
}
var CLASS_ICON="icon";
var CLASS_FA_INFO="fa fa-info-circle";
/** 
 * Adds tootlips to all elements having the class CLASS_ANNOS_NAMES.
 **/
function addTooltips_MetaData() {
    if (tooltips_metadata!= null){
		//find all elements of class CLASS_METADATA
		var metaElements = document.getElementsByClassName(CLASS_METADATA);
		for (var i= 0; i < metaElements.length; i++){
			var tooltip= tooltips_metadata[metaElements[i].innerHTML];
			if (	(tooltip!= null)&&
					(tooltip!= "")){
				metaElements[i].title=tooltip;
				$(metaElements[i]).addClass(CLASS_TOOLTIP);
				//create icon element for info button
				var icon = $( document.createElement('i'));
				icon.addClass(CLASS_FA_INFO);
				icon.addClass(CLASS_ICON);
				$(metaElements[i]).append(icon);
			}
		}
	}
}


function addDescs(){
        $("#structInfoDescription").append("<p>"+structInfoDesc+"</p>");
    
    if (metaDataDesc != null) {
        $("#metaDataDescription").append("<p>"+metaDataDesc+"</p>");
    }
    if (annoDesc != null) {
        $("#annoDescription").append("<p>"+annoDesc+"</p>");
    }
    if (layerDesc != null){
    	for (var i = 0; i < layerDesc.length; i++) {
    		document.getElementById(layerDesc[i].name).innerHTML = layerDesc[i].desc;
    	}
    }
}



/** 
 * Adds tootlips all elements having the class CLASS_METADATA.
 **/
function addTooltips_AnnotationNames() {
    if (tooltips_annonames!= null){
		//find all elements of class CLASS_ANNO_NAMES
		var annoElements = document.getElementsByClassName(CLASS_ANNO_NAMES);
		for (var i= 0; i < annoElements.length; i++){
			var tooltip= tooltips_annonames[annoElements[i].innerHTML];
			if (	(tooltip!= null)&&
					(tooltip!= "")){
				annoElements[i].title=tooltip;
				$(annoElements[i]).addClass(CLASS_TOOLTIP);
				
				//create icon element for info button
				var icon = $(document.createElement('i'));
				icon.addClass(CLASS_FA_INFO);
				icon.addClass(CLASS_ICON);
				icon.addClass(CLASS_ICON);
				icon.addClass(CLASS_TOOLTIP);
				icon.attr('title', tooltip);
				$(annoElements[i]).parent().parent().append(icon);
			}
		}
	}
}

/*******************************************************************************
 * ANNIS link management
 ******************************************************************************/
var CLASS_CLICKIFY = "clickify-anno";
var CLASS_DECLICKIFY = "declickify-anno";


/** Open ANNIS in extra tab or window */
function goANNIS(annoName, annoValue) {
    if ((annisLink != null) &&
        (corpusName != null)) {
        var link = annisLink;
        // add fragment to url
        link = link + "#";

        //create query query (query by the mean of annis, not URI query) part
        if (annoName != null) {
            link = link + "_q=";

            var annoPart = annoName;
            if (annoValue != null) {
                annoPart = annoPart + "=\"" + annoValue + "\"";
            }
            link = link + btoa(annoPart) + "&";
        }
        // create corpus part tu url
        if (corpusName != null) {
		console.log("--------------> corpusName: "+ corpusName);
            link = link + "_c=" + btoa(corpusName);
        }
        //open link in new window
        window.open(link, '_blank');
    }
}

/** Makes a button clickable, means to add class clickify-anno to it **/
function clickifyMe(element) {
    if ((annisLink != null) &&
        (corpusName != null)) {
        element.removeClass(CLASS_DECLICKIFY);
        element.addClass(CLASS_CLICKIFY);
    }
}

/*******************************************************************************
 * Load content for main page
 ******************************************************************************/
function loadMainPage() {
    $('#content')
        .load(
            'main.html',
            function() {
                if (description != null) {
                	$("#corpusDescription").append("<p>"+description+"</p>");
                }
                if (license != null) {
                	$("#license").append("<p>"+license+"</p>");
                }
                if (	(annotators != null)&&
			(annotators.length>0)) {
			console.log("annotators: "+ annotators);
		    var annotatorElement = $("#annotators");
                    for (var i = 0; i < annotators.length; i++) {
                        var span = document.createElement('div');
                        if (annotators[i].eMail != null) {
                        	$(span).append(annotators[i].name + ": <a href='mailto:" + annotators[i].eMail + "'>" + annotators[i].eMail + "<a/>");
                        }
                        else {
                        	$(span).append(annotators[i].name + ": no email available");
                        }
                        $(annotatorElement).append(span);
                    }
                }else{
			console.log("-------------__> EMPTY");
			$("#annotators").css("visibility", "hidden");
		}
            });
}

/*******************************************************************************
 * Load content for impressum page
 ******************************************************************************/
function loadImpressumPage() {
    $('#content')
        .load('impressum.html');
}

/*******************************************************************************
 * Load content for help page
 ******************************************************************************/
function loadHelpPage() {
var helpWindow= window.open('help.html', 'popUpWindow', 'height=530,width=530,left=100,top=100,resizable=no, scrollbars=yes,toolbar=no,menubar=no,location=no,directories=no, status=no,titlebar=no,location=no'); 
helpWindow.focus();
}
