import QtQuick 2.12
import QtQuick.Controls 2.3
import QtQuick.Controls.Styles 1.3
import QtQuick.Extras 1.4
import QtQuick.Layouts 1.3
import QtCharts 2.3
import QtQml 2.12

Item{
    id: root
    property color eum_navy: "#4c5c87";
    property color default_gray: "#efefef";

    property int toolTipTimeout: 3000;
    property int toolTipDelay: 500;

    GridLayout {
        // Source https://stackoverflow.com/questions/34027727/how-can-i-create-a-qml-gridlayout-with-items-of-proportionate-sizes
        id : grid
        x: parent.x
        y: parent.y
        width: parent.width
        height: parent.height
        rows    : 2
        columns : 9
        columnSpacing: 0
        rowSpacing: 0


        property double colMulti : grid.width / grid.columns
        property double rowMulti : grid.height / grid.rows

        function prefWidth(item){
            return colMulti * item.Layout.columnSpan
        }
        function prefHeight(item){
            return rowMulti * item.Layout.rowSpan
        }


        Rectangle {
            id: timestamp_rect
            color : default_gray


            Layout.row       : 0
            Layout.column    : 0
            Layout.rowSpan   : 1
            Layout.columnSpan: 2
            Layout.preferredWidth  : grid.prefWidth(this)
            Layout.preferredHeight : grid.prefHeight(this)
            Layout.bottomMargin: 0;
            Column{
                spacing: 10

                Label{
                    id: current_dt_label
                    text: timebaseModel.currentTimestamp;
                }
            }
        }
        Rectangle {
            id: menu_rect
            property real cboxToButtonRatio: 2/3

            color : default_gray
            border.color: default_gray;

            Layout.row       : 1
            Layout.column    : 0
            Layout.rowSpan   : 1
            Layout.columnSpan: 2
            Layout.preferredWidth  : grid.prefWidth(this)
            Layout.preferredHeight : grid.prefHeight(this)
            Layout.bottomMargin: 0;

            Image {
                id: convFuncPicker
                height:parent.height;
                width: this.height;
                x:0
                y:0
                //anchors.right: dataLayerComboBox.left
                ToolTip.text: qsTr("Select a timebase, based on it's properties");
                ToolTip.visible: convMenuMouseArea.containsMouse
                ToolTip.delay: toolTipDelay
                ToolTip.timeout: toolTipTimeout

                /*
                  TODO(mk):
                            (1) Icons and assorted images to be shipped as file in resources folder,
                                set up qrc for that.
                            (2) DPI scaling for icons? several icon resolutions? Upscaling the icon below
                                leads to a grainy icon.
                */

                source: Qt.resolvedUrl("../data/icons/menu.svg");
                MouseArea{
                    id: convMenuMouseArea
                    anchors.fill: parent;
                    onClicked: convFuncMenu.open();
                    hoverEnabled: true
                }
                Menu {
                    id: convFuncMenu
                    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside;
                    Instantiator{
                        id: convFuncInstantiator
                        model: LayerManager.convFuncModel;
                        delegate: MenuItem{
                            text: model.display
                            onTriggered: {
                                backend.clickConvFuncMenu(model.display);
                            }
                        }
                        onObjectAdded: convFuncMenu.insertItem(index, object);
                        onObjectRemoved: convFuncMenu.removeItem(object);

                    }
                }
            }

            ComboBox{
                id: dataLayerComboBox
                signal reemittedPushedOrPopped;
                signal reemittedTimebaseChanged(var idx);

                anchors {left:convFuncPicker.right ; right: parent.right; }//top: parent.top

                ToolTip.visible: hovered
                ToolTip.delay: toolTipDelay
                ToolTip.timeout: toolTipTimeout
                ToolTip.text: qsTr("Choose a data layer as timebase")


                property var textWidths: []
                property int maxPopupWidth: 300;

                TextMetrics{
                  id: comboBoxTextMetrics
                  text: ""
                }

                textRole: "display"
                model: LayerManager.layerModel

                Component.onCompleted: {
                    currentIndex = 0;
                    LayerManager.layerModel.pushedOrPopped.connect(reemittedPushedOrPopped);
                    backend.didChangeTimebase.connect(reemittedTimebaseChanged);
                    activated.connect(backend.clickComboBoxAtIndex)

                }

                onReemittedTimebaseChanged: {
                    currentIndex = idx;
                }

                onReemittedPushedOrPopped:{
                    textWidths = [];
                    for(var i = 0; i < model.rowCount(); i++) {
                        comboBoxTextMetrics.text = this.model.model[i];
                        let curr_width = comboBoxTextMetrics.width;
                        if (curr_width === NaN){
                            continue;
                        }else{
                            textWidths.push(curr_width);
                        }

                    }
                    let tempMaxWidth = Math.max(textWidths);
                    if((tempMaxWidth !== NaN) && (tempMaxWidth >= maxPopupWidth)){
                        maxPopupWidth = tempMaxWidth;
                        popup.width = maxPopupWidth;
                    }
                    if (dataLayerComboBox.currentIndex === -1){
                        backend.clickComboBoxAtIndex(0);
                        dataLayerComboBox.currentIndex = 0;
                    }
                }

                delegate: ItemDelegate {
                    id: cb_delegate
                    text: model.display
                    width: parent.width
                    property alias fontPointSize: del_text.font.pointSize;

                    // padding controls space between contentItem and backgroundItem
                    topPadding: fontPointSize/4
                    bottomPadding: 0
                    highlighted: dataLayerComboBox.highlightedIndex === index

                    contentItem: Text{
                        id: del_text
                        font.pointSize: 10
                        text: cb_delegate.text
                    }
                    background: Rectangle{
                        color: cb_delegate.highlighted ? "#308cc6":"white"
                        border.color: cb_delegate.highlighted ? Qt.darker("#308cc6"):"white"
                        anchors.fill: cb_delegate
                    }
                }

                popup: Popup {
                    y: 0
                    width: dataLayerComboBox.width
                    implicitHeight: contentItem.implicitHeight//500
                    padding: 1

                    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent

                    contentItem: ListView {
                        clip: true
                        implicitHeight: contentHeight
                        model: dataLayerComboBox.popup.visible ? dataLayerComboBox.delegateModel : null

                        currentIndex: dataLayerComboBox.highlightedIndex
                        spacing: 0
                        ScrollIndicator.vertical: ScrollIndicator { }
                    }
                }
            }

        }
        Rectangle {
            id : timeline_rect
            color : default_gray
            border.color: default_gray;

            Layout.row       : 0
            Layout.column    : 2
            Layout.rowSpan : 2

            Layout.columnSpan : 7
            Layout.preferredWidth  : grid.prefWidth(this)
            Layout.preferredHeight : grid.prefHeight(this)
            Layout.bottomMargin: 0;
            TimelineRuler{}
        }
    }
    FontLoader {
        id: siftFont;
        source: Qt.resolvedUrl("../data/fonts/Andale Mono.ttf");
    }
}
