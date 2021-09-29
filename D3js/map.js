window.onload = () => {

    function generateLeaflet() {
        let mymap = L.map('Leaflet', {}).setView([50, -0.1], 10).setMinZoom(3);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 6,
        }).addTo(mymap);
    }

    function mapGoesBrrrrr() {

        let w = 3000;
        let h = 1250;

        let div = document.getElementById('map');

        let minZoom, maxZoom, countriesGroup, countries, countryLabels, midX, midY, minXY, maxXY, t, zoomWidth, zoomHeight, zoomMidX, zoomMidY, maxXscale, maxYscale, zoomScale, offsetX, offsetY, dleft, dtop;

        let projection = d3
            .geoMercator()
            .center([0, 30])
            .scale([w / (2 * Math.PI)])
            .translate([w / 2, h / 2]);

        let path = d3
            .geoPath()
            .projection(projection);

        function zoomed() {
            t = d3
                .event
                .transform;
            countriesGroup.attr('transform', 'translate(' + [t.x, t.y] + ')scale(' + t.k + ')');
        }

        let zoom = d3
            .zoom()
            .on("zoom", zoomed);


        function initiateZoom() {

            minZoom = Math.max(div.offsetWidth / w, div.offsetHeight / h);
            maxZoom = 20 * minZoom;
            zoom
                .scaleExtent([minZoom, maxZoom])
                .translateExtent([
                    [0, 0],
                    [w, h]
                ]);

            midX = (div.offsetWidth - minZoom * w) / 2;
            midY = (div.offsetHeight - minZoom * h) / 2;

            svg.call(zoom.transform, d3.zoomIdentity.translate(midX, midY).scale(minZoom));

        }

        function boxZoom(box, centroid, paddingPerc) {
            minXY = box[0];
            maxXY = box[1];

            zoomWidth = Math.abs(minXY[0] - maxXY[0]);
            zoomHeight = Math.abs(minXY[1] - maxXY[1]);

            zoomMidX = centroid[0];
            zoomMidY = centroid[1];

            zoomWidth = zoomWidth * (1 + paddingPerc / 100);
            zoomHeight = zoomHeight * (1 + paddingPerc / 100);

            maxXscale = document.getElementById('map_svg').clientWidth / zoomWidth;
            maxYscale = document.getElementById('map_svg').clientHeight / zoomHeight;
            zoomScale = Math.min(maxXscale, maxYscale);

            zoomScale = Math.min(zoomScale, maxZoom);

            zoomScale = Math.max(zoomScale, minZoom);

            offsetX = zoomScale * zoomMidX;
            offsetY = zoomScale * zoomMidY;

            dleft = Math.min(0, document.getElementById('map_svg').clientWidth / 2 - offsetX);
            dtop = Math.min(0, document.getElementById('map_svg').clientHeight / 2 - offsetY);

            dleft = Math.max(document.getElementById('map_svg').clientWidth - w * zoomScale, dleft);
            dtop = Math.max(document.getElementById('map_svg').clientHeight - h * zoomScale, dtop);

            svg
                .transition()
                .duration(500)
                .call(
                    zoom.transform,
                    d3.zoomIdentity.translate(dleft, dtop).scale(zoomScale)
                );

        }

        window.addEventListener('resize', function () {
            svg.attr('width', div.offsetWidth)
                .attr('height', div.offsetHeight);
            initiateZoom();
        })

        let svg = d3
            .select('#map')
            .append('svg')
            .attr('id', 'map_svg')
            .attr('width', div.offsetWidth)
            .attr('height', div.offsetHeight)
            .call(zoom)

        d3.json('https://raw.githubusercontent.com/andybarefoot/andybarefoot-www/master/maps/mapdata/custom50.json', function (json) {
            countriesGroup = svg.append('g').attr('id', 'kaart');
            countriesGroup.append('rect')
                .attr('x', 0)
                .attr('y', 0)
                .attr('width', w)
                .attr('height', h);

            countries = countriesGroup
                .selectAll('path')
                .data(json.features)
                .enter()
                .append('path')
                .attr('d', path)
                .attr('id', function (d, i) {
                    return 'country' + d.properties.iso_a3;
                })
                .attr('class', 'country')
                .on('mouseover', function (d, i) {
                    document.getElementById('country_name').innerHTML = `${d.properties.name}`
                })
                .on('mouseout', function () {
                    document.getElementById('country_name').innerHTML = ``;
                })
                .on('click', function (d, i) {
                    d3.selectAll('.country').classed('country-on', false);
                    d3.select(this).classed('country-on', true);
                    boxZoom(path.bounds(d), path.centroid(d), 20);
                    fetchCountryData(d.properties.iso_a2);
                });

            countryLabels = countriesGroup
                .selectAll('g')
                .data(json.features)
                .enter()
                .append('g')
                .attr('class', 'countrylabel')
                .attr('id', function (d) {
                    return (
                        'translate(' + path.centroid(d)[0] + ',' + path.centroid(d)[1] + ')'
                    );
                })

                .on('click', function (d, i) {
                    d3.selectAll('.country').classed('country-on', false);
                    d3.select('#country' + d.properties.iso_a3).classed('country-on', true);
                    boxZoom(path.bounds(d), path.centroid(d), 20);
                });


            initiateZoom();
        });



    }

    async function fetchCountryData(code) {
        let info, box = document.getElementById('country_detail');
        fetch(`https://api.worldbank.org/v2/country/${code}?format=json`)
            .then(resp => resp.json())
            .then((data) => {
                info = data[1][0];
                box.innerHTML = `
                <p><b>Name:</b> ${info.name} (${info.iso2Code})</p>
                <p><b>Capital:</b> ${info.capitalCity}</p>
                <p><b>Income:</b> ${info.incomeLevel.value}</p>
                <p><b>Coordinates:</b> ${info.latitude}, ${info.longitude}</p>`;
            });

    }

    mapGoesBrrrrr();
    generateLeaflet();
}