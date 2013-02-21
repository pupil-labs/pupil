import svgwrite

svg_document = svgwrite.Drawing(filename = "test-svgwrite.svg",
                                size = ("800px", "600px"))

svg_document.add(svgwrite.shapes.Polyline([(100,100),(200,130),(300,20)], fill='none',stroke='black'))


print(svg_document.tostring())

svg_document.save()