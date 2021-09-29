window.onload = () => {
    function generateMap() {
        d3.json('./data/countries_visited.json').then(function (data) {
            let width = 200;
            let height = 200;
            let projection = d3.geoMercator();
            projection.fitSize([width, height], data);
            let geoGenerator = d3.geoPath().projection(projection);

            let svg = d3.select('content').append('svg').attr('width', width).attr('height', height).attr('viewbox', [0, 0, width, height]);

            svg.append('g').enter().selectAll('path').data(data.features).append('path').attr('d', geoGenerator).attr('fill', '#088').attr('stroke', '#000');
            svg.node();
            console.log('done')
            console.log(data.features)
        });
    }

  
    generateMap();
}