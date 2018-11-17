#include <CGAL/Simple_cartesian.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/random_polygon_2.h>
#include <CGAL/Random.h>
#include <CGAL/algorithm.h>
#ifdef CGAL_USE_GMP
#include <CGAL/Gmpz.h>
typedef CGAL::Gmpz RT;
#else
// NOTE: the choice of double here for a number type may cause problems
//       for degenerate point sets
#include <CGAL/double.h>
typedef double RT;
#endif
#include <fstream>
#include <list>
typedef CGAL::Simple_cartesian<RT>                        K;
typedef K::Point_2                                        Point_2;
typedef std::list<Point_2>                                Container;
typedef CGAL::Polygon_2<K, Container>                     Polygon_2;
typedef CGAL::Creator_uniform_2<int, Point_2>             Creator;
typedef CGAL::Random_points_in_square_2<Point_2, Creator> Point_generator;
int main(int argc, char** argv)
{
   if(argc < 3){
      std::cout << "Not enough arguments" << std::endl;
      return 0;
   }
   double RADIUS;
   int MAX_POLY_SIZE; 
   sscanf(argv[1],"%lf", &RADIUS);
   MAX_POLY_SIZE = atoi(argv[2]);

   Polygon_2            polygon;
   std::list<Point_2>   point_set;
   CGAL::Random         rand;

   int size = rand.get_int(4, MAX_POLY_SIZE);
   CGAL::copy_n_unique(Point_generator(RADIUS), size,
                       std::back_inserter(point_set));

   CGAL::random_polygon_2(point_set.size(), std::back_inserter(polygon),
                          point_set.begin());
   std::cout << polygon;
   return 0;
}
