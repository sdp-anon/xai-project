import java.util.*;

public class OrderProcessor {

    private List<String> orders = new ArrayList<>();

    public void addOrder(String order) {
        orders.add(order);
    }

    public void processOrders() {
        for (String order : orders) {
            if (order == null) {
                continue;
            }

            if (order.length() > 5) {
                validate(order);
                ship(order);
            } else {
                cancel(order);
            }
        }
    }

    private void validate(String order) {
        if (!order.contains("VALID")) {
            throw new RuntimeException("Invalid order");
        }
    }

    private void ship(String order) {
        System.out.println("Shipping: " + order);
    }

    private void cancel(String order) {
        System.out.println("Cancelled: " + order);
    }
}