let values = [];
let x, y, lastmonth, svg, graph;
let url = (x, y, z) => `https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/${z}/${x}/${y}${devicePixelRatio > 1 ? "@2x" : ""}?access_token=pk.eyJ1IjoibWJvc3RvY2siLCJhIjoiY2s5ZWRlbTM4MDE0eDNocWJ2aXR2amNmeiJ9.LEyjnNDr_BrxRmI4UDyJAQ`
const width = window.innerWidth - 150;
const height = 300;
const margin = {
    top: 50,
    bottom: 50,
    left: 50,
    right: 50
};


window.onload = () => {
    loadData();
}

async function loadData() {
    await d3.csv('./data/temperature.csv', function (data) {
        data.temperature = Math.floor(data.temperature);
        values.push(data);
    });

    createGraph();

    await fetch('./data/miserables.json').then(res => res.json()).then(json => generateForceGraph(json));
}

function createGraph() {
    const svg = d3.select('#graph')
        .append('svg')
        .attr('height', height)
        .attr('width', width)
        .attr('viewbox', [0, 0, width, height]);

    x = d3.scaleBand()
        .domain(d3.range(values.length))
        .range([margin.left, width - margin.right])
        .padding(0.1);

    y = d3.scaleLinear()
        .domain([0, 100])
        .range([height - margin.bottom, margin.top]);

    svg.append('g')
        .attr('fill', 'royalblue')
        .selectAll('rect')
        .data(values)
        .join('rect')
        .attr('x', (d, i) => x(i))
        .attr('y', (d) => y(d.temperature))
        .attr('height', d => y(0) - y(d.temperature))
        .attr('width', x.bandwidth())
        .attr('class', 'rect')

    svg.append('g').call(xAxis);
    svg.append('g').call(yAxis);
    svg.node();

}

function xAxis(g) {
    g
        .attr('transform', `translate(0, ${250})`)
        .call(d3.axisBottom(x).tickFormat((i) => {
            let a = String(values[i].date).split('-');
            if (a[1] != lastmonth) {
                lastmonth = a[1];
                return a[1];
            }

        }))
        .attr('font-size', '10px');
}

function yAxis(g) {
    g.attr('transform', `translate(${margin.left}, 0 )`)
        .call(d3.axisLeft(y).ticks(null, values.format))
}

function generateForceGraph(data) {
    graph = ForceGraph()(document.getElementById('chart')).graphData(data).width(window.innerWidth - 100).height(500).nodeAutoColorBy('group').nodeLabel('id');
}


