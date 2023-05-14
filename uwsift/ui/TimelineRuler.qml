import QtQml 2.12
import QtQuick 2.12
import QtQuick.Controls 2.3

/*
Blueprint:
    id
    property declarations
    signal declarations
    JavaScript functions
    object properties
    child objects
    states
    transitions

*/
Rectangle{
    id: timelineRuler

    property date minDate: new Date("2011-01-01")
    property date maxDate: new Date("2030-12-31")
    property date oldDate: new Date("2011-01-01")

    signal siftTimelinePosChanged(real xPosition, real yPosition)

    anchors.fill: parent
    border.color: Qt.rgba(1, 0, 0, 0)//default_gray
    color: Qt.rgba(0, 0, 0, 0)

    Canvas{
        // ID
        id: timelineRulerCanvas
        // Property declarations
        property date oldDate: new Date("1990-01-01");
        property real tickWidth;
        property int maxNumTicks: 10;
        property int temp_idx: -1;
        property var tickBlueprints: [];
        property var tickDates: [];
        property var tickMargin: 10;
        property var resolution: 30;
        property var resolutionMode: "Minutes";
        property var majorTickFontsize: 14;
        property var tickFontSize: 14;
        // textMargin chosen to be 90 based on font and fontsize
        // TODO(mk): attach signal/event handler to tickFontSize to recalculate textMargin
        property var textMargin: 90;
        property var rulerYPosition: timestamp_rect.height + 0.5;
        // Signal declarations
        // Javascript functions
        function clear_canvas(context) {
            context.reset();
        }

        function debugCanvas(context){
            context.strokeStyle = 'black';
            context.lineWidth = 5;
            context.beginPath();
            context.moveTo(0,0);
            context.lineTo(width, height);
            context.stroke();
            context.lineWidth = 1;
        }

        function nextDateByResolution(date, resolution, resolutionMode){
            let retDate = new Date(date);
            if (resolutionMode === "Minutes"){
                retDate.setMinutes(date.getMinutes()+resolution);
            }else if(resolutionMode === "Hours"){
                retDate.setHours(date.getHours()+resolution);
            }else if(resolutionMode === "Days"){
                retDate.setDays(date.getDays()+resolution);
            }
            return retDate;
        }

        function previousDateByResolution(date, resolution, resolutionMode){
            return nextDateByResolution(date, -resolution, resolutionMode);
        }

        function buildTickBlueprints(){
            timelineRulerCanvas.tickBlueprints = []

            // Assume temporally sorted timebaseModel
            let numDts = timebaseModel.rowCount()
            let currMinDate = timebaseModel.at(0);
            let currMaxDate = timebaseModel.at(numDts-1);
            let firstDate = new Date(currMinDate)
            firstDate.setSeconds(0);
            let secondDate = nextDateByResolution(firstDate, resolution, resolutionMode);
            let resolutionSeconds = (secondDate.getTime() - firstDate.getTime())/1000;
            let dataTimeSpanSeconds = (currMaxDate.getTime()-currMinDate.getTime())/1000;
            let maxTickIndex = Math.ceil(dataTimeSpanSeconds/resolutionSeconds);
            // If only one timestep is loaded draw 2 ticks regardless.
            if (maxTickIndex === 0){
                maxTickIndex = 1;
            }
            timelineRulerCanvas.tickWidth = ((timelineRulerCanvas.width-timelineRulerCanvas.textMargin) / maxTickIndex);
            let tickDts = [firstDate];
            for(var k=1; k<=maxTickIndex; k++){
                let nextTickDt = nextDateByResolution(tickDts[k-1], resolution, resolutionMode);
                tickDts.push(nextTickDt);

            }

            timelineRulerCanvas.tickDates = tickDts;

            // figure out temporal res. and scale accordingly
            let ticks = timelineRulerCanvas.tickDates
            for(var i=0; i < ticks.length; i++){
                let tickBP = {"X": 1.0,"Y":1.0,"TextX":1.0,"TextY":1.0,"Length":1.0,"Major":false, "Text":""};
                // major or minor tick
                let dCurr = ticks[i];
                let dCurrDay = dCurr.getDate();
                let dPrev;
                if (i === 0){
                    dPrev = previousDateByResolution(dCurr, resolution, resolutionMode);
                }else{
                    dPrev = ticks[i-1];
                }
                let dprevDay = dPrev.getDate();
                if ((dPrev < dCurr) && (dprevDay < dCurrDay)){
                    // Major tick
                    // Event handlers (such as onWidthChanged may be called more than once,
                    // thus no mutating global state
                    tickBP = createTickBlueprint(i, dCurr, true);
                }else{
                    tickBP = createTickBlueprint(i, dCurr, false);
                }
                timelineRulerCanvas.tickBlueprints.push(tickBP);
            }
        }

        function createTickBlueprint(index, tickDate, majorTick){
            let tickBP = {"X": 1.0,"Y":1.0,"TextX":1.0,"TextY":1.0,"Length":1.0,"Major":false, "Text":"", "MajorText":"","MajorTextY": timelineRulerCanvas.majorTickFontsize};
            if (majorTick || (index === 0)){
                tickBP.Major = true;
                tickBP.MajorText += Qt.formatDateTime(tickDate, "d MMM yyyy");
                tickBP.Y = 0;
                tickBP.Length = height;
            }else{
                // Minor tick
                tickBP.Y = height/4;
                tickBP.Length = height;
            }
            tickBP.X = Math.round(timelineRulerCanvas.tickMargin + index*tickWidth) + 0.5;
            tickBP.Text += Qt.formatDateTime(tickDate, "hh:mm");
            tickBP.TextX = tickBP.X + 2;
            tickBP.TextY = rulerYPosition - 1;
            return tickBP;
        }

        function drawTicks(context){
            // Structure of tickBP:
            //              {"X": 1.0,"Y":1.0,"TextX":1.0,"TextY":1.0,"Length":1.0,"Major":false, "Text":""};
            // Font size in px
            let fontSize = 16;
            let textPaddingBottom = 3;
            context.font= tickFontSize+'px "%1"'.arg("sans-serif");//.arg(siftFont.name);
            // TODO: the below does not work, as a child object would be necessary
            //context.font= fontSize+'px "%1"'.arg(root.siftFont.name);
            //       Solution: Create a Resources.qml (or any compatible name, must start with Capital letter)
            //                 and include it as a child object wherever resources of some kind are needed.

            timelineRulerCanvas.tickBlueprints.forEach((item, index) => {
                context.beginPath();
                context.moveTo(item.X, item.Y);
                context.lineTo(item.X, item.Y + item.Length);
                if (item.Major){
                    let majorTickText = item.MajorText;
                    context.fillText(majorTickText, item.TextX, item.MajorTextY);
                    context.fillText(item.Text, item.TextX, item.TextY-textPaddingBottom);
                }else{
                    context.fillText(item.Text, item.TextX, item.TextY-textPaddingBottom);
                }

                context.stroke();
            });
        }

        // Object properties
        anchors.fill: parent
        renderStrategy: Canvas.Threaded

        onPaint:{
            var context = getContext("2d");
            context.reset()
            context.strokeStyle = Qt.darker(timeline_rect.color, 2.0)
            context.lineWidth = 1;
            // Draw horizontal ray
            context.moveTo(0, rulerYPosition);
            context.lineTo(width, rulerYPosition);
            context.stroke();
            let margin = 10;
            drawTicks(context);
        }
        /*
            backend holds resolution: minutes, hours, etc....
            ruler gets drawn until timeline is full
                - redraw ruler to region of interest on data load
                    - draw one tick before/after data time if maxNumTicks allows this?
                        -> what are the different modes regarding too many/very few ticks that can occur?
            on data load, chevron markers are drawn at points on timeline where data occurs
            on anim outlined chevron (with whisker?) moves over these points
        */

        function calculate_resolution(){
            let possibleResolutions = [5,15,30,60,90,120];
            let numDts =  timebaseModel.rowCount();
            let totalDeltaDt = timebaseModel.at(numDts-1).getTime() - timebaseModel.at(0).getTime();
        }

        Connections {
            target: timebaseModel
            function onTimebaseChanged() {
                timelineRulerCanvas.calculate_resolution();
                timelineRulerCanvas.buildTickBlueprints();
                timelineRulerCanvas.requestPaint();
            }
        }
        onWidthChanged: {
            buildTickBlueprints();
            requestPaint();
        }
        FontLoader {
            id: siftFont;//"Sans Serif"
            source: Qt.resolvedUrl("../data/fonts/Andale Mono.ttf");//"qrc:/AndaleMono.ttf"
        }
        /*
            child objects
            states
            transitions
        */
    }
    Canvas{
        // Id
        id: timelineMarkerCanvas;

        // Property declarations
        property var currIndex: 0
        property var cursorX: 0
        property var cursorY: 0
        // marker radius actually used to define bounding box of rounded rect
        property var markerRadius: 10
        property var markerYPosition: (3/4)*(height) - (markerRadius / 2);
        property bool dataLoaded: false;
        property var markerBluePrints: []

        // signals
        signal reemittedTimelineIndexChanged(var idx);
        signal reemittedRefreshTimeline();
        // JS functions
        function calculateTickWidthPerTime(ticks){
            let tickWidthPerTime;
            if (ticks.length === 1){
                let nextTick = timelineRulerCanvas.nextDateByResolution(ticks[0], timelineRulerCanvas.resolution, timelineRulerCanvas.resolutionMode);
                tickWidthPerTime = timelineRulerCanvas.tickWidth/(nextTick.getTime()-ticks[0].getTime());
            }else{
                tickWidthPerTime = timelineRulerCanvas.tickWidth/(ticks[1].getTime()-ticks[0].getTime());
            }
            return tickWidthPerTime;
        }

        function updateTimelineCursorPosition(){
            let resolution = timelineRulerCanvas.resolution;
            let resolutionMode = timelineRulerCanvas.resolutionMode;
            let ticks = timelineRulerCanvas.tickDates;
            let tickWidthPerTime = calculateTickWidthPerTime(ticks);
            let markerDate = timebaseModel.at(currIndex);
            let timeOffset = markerDate.getTime()-ticks[0].getTime();
            let markerWidth = timelineMarkerCanvas.markerBluePrints[0].W;
            timelineMarkerCanvas.cursorX = (tickWidthPerTime*timeOffset) + timelineRulerCanvas.tickMargin;
            timelineMarkerCanvas.cursorY = timelineRulerCanvas.rulerYPosition;
        }

        function createMarkerBlueprint(index, markerDate){
            let markerBP = {"X": 1.0,"Y":1.0,"W": 1.0, "H":1.0, "R": 1.0};
            markerBP.W = markerRadius;
            markerBP.H = markerBP.W;
            markerBP.R = markerBP.W * 0.5;

            let resolution = timelineRulerCanvas.resolution;
            let resolutionMode = timelineRulerCanvas.resolutionMode;
            let ticks = timelineRulerCanvas.tickDates;
            let tickWidthPerTime = calculateTickWidthPerTime(ticks);
            let timeOffset = markerDate.getTime()-ticks[0].getTime();
            markerBP.X = (tickWidthPerTime*timeOffset) + timelineRulerCanvas.tickMargin - (markerBP.W/2);
            markerBP.Y = markerYPosition;

            return markerBP;
        }

        function updateMarkerBlueprints(){
            timelineMarkerCanvas.markerBluePrints = [];

            for(var i=0; i < timebaseModel.rowCount(); i++){
                timelineMarkerCanvas.markerBluePrints.push(createMarkerBlueprint(i, timebaseModel.at(i)));
            }
        }

        // object properties
        anchors.fill: parent;
        renderStrategy: Canvas.Threaded

        onPaint: {
            var context = getContext("2d");
            context.reset();

            if (dataLoaded){
                updateTimelineCursorPosition()
                let markerColor = Qt.darker(Qt.rgba(1, 0, 0, 1), 1);
                context.strokeStyle = markerColor;
                context.fillStyle = markerColor;
                context.lineWidth = 2;
                markerBluePrints.forEach((bp)=>{
                    context.beginPath();
                    context.roundedRect(bp.X, bp.Y, bp.W, bp.H, bp.R, bp.R);
                    context.stroke();
                });
                context.beginPath();

                // draw Cursor
                let cursorWidth = 11
                let cursorYOffset = Math.round(cursorWidth/2)-1
                context.moveTo(cursorX, cursorY-cursorYOffset);
                context.lineTo(cursorX, markerYPosition+1.5*markerRadius);
                context.fillRect(cursorX-5, cursorY-cursorYOffset, cursorWidth, cursorWidth);
                context.stroke();
            }

        }

        onReemittedTimelineIndexChanged: {
            currIndex = idx;
            requestPaint();
        }
        onReemittedRefreshTimeline: {
            timelineMarkerCanvas.updateMarkerBlueprints();
            timelineMarkerCanvas.requestPaint();
        }
        // Connections
        Connections{
            target: timelineRulerCanvas
            function onWidthChanged() {
                timelineMarkerCanvas.updateMarkerBlueprints();
                timelineMarkerCanvas.requestPaint();
            }
        }
        Connections{
            target: timebaseModel
            function onTimebaseChanged() {
                timelineMarkerCanvas.dataLoaded = true
                timelineMarkerCanvas.currIndex = 0;
                timelineMarkerCanvas.updateMarkerBlueprints();
                timelineMarkerCanvas.requestPaint();
            }
        }

        Connections{
            target: timebaseModel
            function onReturnToInitialState() {
               timelineMarkerCanvas.dataLoaded = false;
            }
        }

        Component.onCompleted: {
            backend.doNotifyTimelineIndexChanged.connect(reemittedTimelineIndexChanged);
            backend.doRefreshTimeline.connect(reemittedRefreshTimeline);
        }
    }
    MouseArea{
        id: mouse_area
        property var prevIndex: 0;

        hoverEnabled: true
        // add some more height to allow clicks in the area of marker canvas
        // being handled
        anchors.fill: parent;

        onClicked: {
            if (!timelineMarkerCanvas.dataLoaded)
                return;

            let markerCenterXs = []
            for (var i=0; i<timelineMarkerCanvas.markerBluePrints.length;++i){
                let markerCenterX =
                    timelineMarkerCanvas.markerBluePrints[i].X + timelineMarkerCanvas.markerBluePrints[i].R;
                markerCenterXs.push(markerCenterX);
            }

            let distVals = markerCenterXs.map((markerCenterX)=>{
                return Math.abs(mouseX - markerCenterX)
            });
            let val = Math.min(...distVals);
            let clickedIndex = distVals.indexOf(val)

            backend.clickTimelineAtIndex(clickedIndex);
            timelineMarkerCanvas.currIndex = clickedIndex;
            timelineMarkerCanvas.requestPaint();
        }
    }
}
