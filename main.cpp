#include <QCoreApplication>
#include <QCommandLineParser>

#include <iostream>

#include "pathtracer.h"
#include "scene/scene.h"

#include <QImage>

#define IMAGE_WIDTH 100
#define IMAGE_HEIGHT 100

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addPositionalArgument("scene", "Scene file to be rendered");
    parser.addPositionalArgument("output", "Image file to write the rendered image to");

    if(!parser.parse(QCoreApplication::arguments())) {
        std::cerr << parser.errorText().toStdString() << std::endl;
        a.exit(1);
        return 1;
    }

    const QStringList args = parser.positionalArguments();
    if(args.size() != 2) {
        std::cerr << "Error: Wrong number of arguments" << std::endl;
        a.exit(1);
        return 1;
    }
    QString scenefile = args[0];
    QString output = args[1];

    QImage image(IMAGE_WIDTH, IMAGE_HEIGHT, QImage::Format_RGB32);

    Scene scene = Scene::load(scenefile);

    PathTracer tracer(IMAGE_WIDTH, IMAGE_HEIGHT);

    QRgb *data = reinterpret_cast<QRgb *>(image.bits());

    tracer.traceScene(data, scene);

    bool success = image.save(output);
    if(success) {
        std::cout << "Wrote rendered image to " << output.toStdString() << std::endl;
    }
    a.exit();
}
